import numpy as np
from matplotlib.pyplot import get_cmap
from imgaug.augmentables.bbs import BoundingBox

cm = get_cmap('gist_rainbow')


class Visualizer:
    def __init__(self, classes, score_threshold, mean, std, alpha=0.5):
        self.classes = classes
        self.score_threshold = score_threshold
        self.mean = mean
        self.std = std
        self.alpha = 0.5
        self.cmap = [
            [int(y * 255.0) for y in cm(1.0 * x / len(self.classes))[: 3]]
            for x in range(len(self.classes))]

    def visualize_detections(
            self, image, pred_boxes, pred_classes, pred_scores, gt_boxes,
            gt_classes):

        pred_img = (
            (image *
             self.std +
             self.mean) *
            255).astype(
            np.uint8).copy()
        gt_img = pred_img.copy()

        for i in range(gt_boxes.shape[0]):
            cid = int(gt_classes[i])
            bb = BoundingBox(*gt_boxes[i])
            bb.label = self.classes[cid]['name']
            gt_img = bb.draw_on_image(gt_img, self.cmap[cid], self.alpha, 2)

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            cid = int(pred_classes[i])
            bb = BoundingBox(*pred_boxes[i])
            bb.label = f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}"
            pred_img = bb.draw_on_image(
                pred_img, self.cmap[cid], self.alpha, 2)

        result = np.hstack([pred_img, gt_img])
        return result.transpose(2, 0, 1)
