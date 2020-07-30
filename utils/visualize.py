import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Visualizer:
    def __init__(self, score_threshold, mean, std, alpha=0.5):
        self.score_threshold = score_threshold
        self.mean = mean
        self.std = std
        self.alpha = 0.5

    def visualize_detections(
            self, image, pred_boxes, pred_classes, pred_scores, gt_boxes,
            gt_classes):

        gt = []
        pred = []

        for i in range(gt_boxes.shape[0]):
            bb = BoundingBox(*gt_boxes[i])
            bb.label = gt_classes[i]
            gt.append(bb)

        for i in range(pred_boxes.shape[0]):
            bb = BoundingBox(*pred_boxes[i])
            bb.label = f"{pred_classes[i]}: {pred_scores[i]:.2f}"
            pred.append(bb)

        image = ((image * self.std + self.mean) * 255).astype(np.uint8)
        gt_bbs = BoundingBoxesOnImage(
            gt, image.shape).draw_on_image(
            image, size=2, alpha=self.alpha)
        pred_bbs = BoundingBoxesOnImage(pred, image.shape).draw_on_image(
            image, size=2, alpha=self.alpha)

        result = np.hstack([pred_bbs, gt_bbs])
        return result.transpose(2, 0, 1)
