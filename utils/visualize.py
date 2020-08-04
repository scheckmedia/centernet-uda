import numpy as np
from matplotlib.pyplot import get_cmap
from imgaug.augmentables.bbs import BoundingBox

cm = get_cmap('gist_rainbow')


class Visualizer:
    def __init__(self, classes, score_threshold,
                 mean, std, font_size=14, alpha=0.5):
        self.classes = classes
        self.score_threshold = score_threshold
        self.font_size = font_size
        self.mean = mean
        self.std = std
        self.alpha = alpha
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
            gt_img = bb.draw_box_on_image(
                gt_img, self.cmap[cid], self.alpha, 2)
            gt_img = bb.draw_label_on_image(
                gt_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            cid = int(pred_classes[i])
            bb = BoundingBox(*pred_boxes[i])
            bb.label = f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}"
            pred_img = bb.draw_box_on_image(
                pred_img, self.cmap[cid], self.alpha, 2)
            pred_img = bb.draw_label_on_image(
                pred_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        result = np.hstack([pred_img, gt_img])
        return result.transpose(2, 0, 1)
