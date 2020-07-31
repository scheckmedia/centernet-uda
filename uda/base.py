import logging

from pathlib import Path
import numpy as np
import torch

from models.decode import decode_detection

log = logging.getLogger("uda")


class UDA():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.centernet_loss = None
        self.device = None
        self.summary_writer = None
        self.classes = None
        self.visualizer = None

        super().__init__()

    def step(self, data, is_training=True):
        for k in data:
            data[k] = data[k].to(device=self.device, non_blocking=True)

        if is_training:
            self.optimizer.zero_grad()

        outputs_source_domain = self.model(data["input"])
        outputs_target_domain = self.model(data["target_domain_input"])

        outputs = {
            "source_domain": outputs_source_domain,
            "target_domain": outputs_target_domain
        }
        loss, stats = self.criterion(outputs, data)

        if is_training:
            loss.backward()
            self.optimizer.step()

        stats["total_loss"] = loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats

        return outputs

    def get_detections(self, outputs, batch):
        src = outputs["source_domain"]

        dets = decode_detection(src["hm"], src["wh"], src["reg"])
        dets = dets.detach().cpu().numpy()
        dets[:, :, :4] *= self.model.down_ratio

        ids = batch["id"].cpu().numpy()
        mask = (batch["reg_mask"].detach().cpu().numpy() == 1).squeeze()
        dets_gt = batch["gt_dets"].cpu().numpy()
        areas_gt = batch["gt_areas"].cpu().numpy()
        dets_gt[:, :, :4] *= self.model.down_ratio

        gt_boxes = []
        gt_clss = []
        gt_ids = []
        gt_areas = []
        for i in range(dets_gt.shape[0]):
            det_gt = dets_gt[i, mask[i]]

            gt_boxes.append(det_gt[:, :4])
            gt_clss.append(det_gt[:, 5].astype(np.int32))
            gt_ids.append(ids[i])
            gt_areas.append(areas_gt[i, mask[i]])

        return {
            'pred_boxes': dets[:, :, :4],
            'pred_classes': dets[:, :, 5].astype(np.int32),
            'pred_scores': dets[:, :, 4],
            'gt_boxes': gt_boxes,
            'gt_classes': gt_clss,
            'gt_ids': gt_ids,
            'gt_areas': gt_areas
        }

    def criterion(self, outputs, batch):
        raise NotImplementedError

    def load_model(self, path, resume=False):
        path = Path(path)
        if not path.exists():
            log.warning(f"Model path {path} does not exists!")
            return 1

        checkpoint = torch.load(path)
        epoch = checkpoint["epoch"]
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]

        model_state_dict = self.model.state_dict()

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    log.warning(
                        f"skip parameter {k} because of shape mismatch")
                    state_dict[k] = model_state_dict[k]
            else:
                log.info(f"drop parameter {k}")

        for k in model_state_dict:
            if k not in state_dict:
                log.warning(f"no parameter {k} available")
                state_dict[k] = model_state_dict[k]

        self.model.load_state_dict(state_dict, strict=False)
        return epoch if resume else 1

    def save_model(self, path, epoch, with_optimizer=False):
        state_dict = self.model.state_dict()

        data = {
            'epoch': epoch,
            'state_dict': state_dict
        }
        if with_optimizer:
            data["optimizer"] = self.optimizer.state_dict()

        torch.save(data, path)

    def log_detections(self, batch, detections, step, tag):
        images = batch["input"].detach().cpu().numpy()

        for i in range(images.shape[0]):
            result = self.visualizer.visualize_detections(
                images[i].transpose(1, 2, 0),
                detections['pred_boxes'][i],
                [self.classes[int(x)]['name']
                 for x in detections['pred_classes'][i]],
                detections['pred_scores'][i],
                detections['gt_boxes'][i],
                [self.classes[int(x)]['name']
                 for x in detections['gt_classes'][i]])
            self.summary_writer.add_image(f'{tag}/detections', result, step)

    def log_stat(self, name, value, step):
        self.summary_writer.add_scalar(name, value, step)
