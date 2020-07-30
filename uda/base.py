import torch
from models.decode import decode_detection
import numpy as np


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

    def log_detections(self, batch, detections, step, tag='validation'):
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

    def log_stats(self, stats, step):
        for k, v in stats.items():
            self.summary_writer.add_scalar(k, v.avg, step)
            v.reset()
