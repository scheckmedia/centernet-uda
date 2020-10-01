import numpy as np
from matplotlib.pyplot import get_cmap
from imgaug.augmentables import BoundingBox, Polygon
from utils.box import rotate_bbox
import cv2

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

        if gt_boxes.shape[-1] == 5:
            return self.__draw_rotated_box(
                gt_img, gt_boxes, gt_classes, pred_img, pred_boxes,
                pred_scores, pred_classes)

        return self.__draw_box(gt_img, gt_boxes, gt_classes,
                               pred_img, pred_boxes, pred_scores, pred_classes)

    def __draw_box(self, gt_img, gt_boxes, gt_classes,
                   pred_img, pred_boxes, pred_scores, pred_classes):
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

    def __draw_rotated_box(self, gt_img, gt_boxes, gt_classes,
                           pred_img, pred_boxes, pred_scores, pred_classes):

        for i in range(gt_boxes.shape[0]):
            cid = int(gt_classes[i])
            rot_pts = np.array(rotate_bbox(*gt_boxes[i]))
            contours = np.array(
                [rot_pts[0],
                 rot_pts[1],
                 rot_pts[2],
                 rot_pts[3]])
            # bb.label = self.classes[cid]['name']
            cv2.polylines(
                gt_img,
                [contours],
                isClosed=True,
                color=self.cmap[cid],
                thickness=2,
                lineType=cv2.LINE_4)
            poly = Polygon(
                np.array(contours, dtype=np.int32).reshape(-1, 2),
                label=self.classes[cid]['name'])

            gt_img = poly.to_bounding_box().draw_label_on_image(
                gt_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        for i in range(pred_boxes.shape[0]):
            if pred_scores[i] < self.score_threshold:
                continue

            cid = int(pred_classes[i])
            rot_pts = np.array(rotate_bbox(*pred_boxes[i]))
            contours = np.array(
                [rot_pts[0],
                 rot_pts[1],
                 rot_pts[2],
                 rot_pts[3]])

            # bb.label = f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}"
            poly = Polygon(
                np.array(contours, dtype=np.int32).reshape(-1, 2),
                label=f"{self.classes[int(cid)]['name']}: {pred_scores[i]:.2f}")

            cv2.polylines(
                pred_img,
                [contours],
                isClosed=True,
                color=self.cmap[cid],
                thickness=2,
                lineType=cv2.LINE_4)

            pred_img = poly.to_bounding_box().draw_label_on_image(
                pred_img,
                self.cmap[cid],
                size_text=self.font_size,
                alpha=self.alpha,
                height=self.font_size + 4)

        result = np.hstack([pred_img, gt_img])
        return result.transpose(2, 0, 1)
