from torch.utils import data
from pycocotools.coco import COCO
from glob import glob
import hydra
import numpy as np
from omegaconf.listconfig import ListConfig
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image
from utils.image import draw_umich_gaussian as draw_gaussian, gaussian_radius

import logging
from pathlib import Path
log = logging.getLogger(__name__)


class Dataset(data.Dataset):
    def __init__(self, image_folder, annotation_file,
                 input_size=(512, 512), target_domain_glob=None, num_classes=80,
                 mean=(0.40789654, 0.44719302, 0.47026115), std=(0.28863828, 0.27408164, 0.27809835),
                 augmentation=None, max_detections=150, down_ratio=4):
        self.image_folder = Path(image_folder)
        self.coco = COCO(annotation_file)
        self.images = self.coco.getImgIds()
        self.max_detections = max_detections
        self.down_ratio = down_ratio
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.num_classes = num_classes
        self.cat_mapping = {v: i for i,
                            v in enumerate(range(1, num_classes + 1))}

        assert len(input_size) == 2

        if isinstance(target_domain_glob, str):
            self.target_domain_files = glob(target_domain_glob)
        elif isinstance(target_domain_glob, (list, ListConfig)):
            self.target_domain_files = []
            for pattern in target_domain_glob:
                self.target_domain_files.extend(glob(pattern))

        if augmentation:
            augmentation_methods = []
            for augment in augmentation:
                method = list(augment)[0]
                params = dict(augment[method])

                for k, v in params.items():
                    if isinstance(v, (list, ListConfig)):
                        params[k] = tuple(v)
                m = hydra.utils.get_method(
                    f"imgaug.augmenters.{method}")(**params)
                augmentation_methods.append(m)

                log.debug(
                    f"Register imgaug.augmenters.{method} as augmentation method")

            self.augmentation = iaa.Sequential(augmentation_methods)

        self.resize = iaa.Resize((self.input_size[0], self.input_size[1]))
        self.resize_out = iaa.Resize(
            (self.input_size[0] // down_ratio, self.input_size[1] // down_ratio))

        log.info(
            f"found {len(self.target_domain_files)} samples for target domain")
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = self.image_folder / file_name
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_detections)

        img = np.array(Image.open(img_path).convert("RGB"))

        boxes = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            boxes.append(BoundingBox(*bbox))

        bbs = BoundingBoxesOnImage(boxes, shape=img.shape)

        if self.augmentation is not None:
            img_aug, bbs_aug = self.augmentation(image=img, bounding_boxes=bbs)
        else:
            img_aug, bbs_aug = np.copy(img), bbs.copy()

        img_aug, bbs_aug = self.resize(image=img_aug, bounding_boxes=bbs_aug)

        img = (img_aug.astype(np.float32) / 255.)
        inp = (img - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = self.input_size[1] // self.down_ratio
        output_w = self.input_size[0] // self.down_ratio
        num_classes = self.num_classes

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_detections, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_detections, 2), dtype=np.float32)
        ind = np.zeros((self.max_detections), dtype=np.int64)
        reg_mask = np.zeros((self.max_detections), dtype=np.uint8)
        cat_spec_wh = np.zeros(
            (self.max_detections, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros(
            (self.max_detections, num_classes * 2), dtype=np.uint8)

        bbs_aug = self.resize_out(bounding_boxes=bbs_aug)
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox_aug = bbs_aug[k].clip_out_of_image((output_w, output_h))
            bbox = np.array([bbox_aug.x1, bbox_aug.y1,
                             bbox_aug.x2, bbox_aug.y2])

            cls_id = int(self.cat_mapping[ann['category_id']])

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((np.ceil(h), np.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        del bbs
        del bbs_aug
        del img_aug

        target_domain_img = np.array(Image.open(
            np.random.choice(self.target_domain_files)).convert("RGB"))
        target_domain_img = self.resize(image=target_domain_img)
        target_domain_img = np.array(
            target_domain_img, dtype=np.float32) / 255.0
        target_domain_img = (target_domain_img - self.mean) / self.std
        target_domain_img = target_domain_img.transpose(2, 0, 1)

        ret = {'input': inp, 'hm': hm, 'target_domain_input': target_domain_img,
               'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg}

        return ret

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox
