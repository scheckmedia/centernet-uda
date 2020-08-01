from tensorboardX import SummaryWriter
from utils.visualize import Visualizer


class TensorboardLogger:
    def __init__(self, cfg, classes):
        super().__init__()
        self.classes = classes
        self.summary_writer = SummaryWriter('logs')
        self.visualizer = Visualizer(
            cfg.tensorboard.score_threhsold,
            cfg.normalize.mean,
            cfg.normalize.std)
        self.num_visualizations = cfg.tensorboard.num_visualizations
        self.__num_logged_images = 0

    def log_detections(self, batch, detections, step, tag):
        if self.__num_logged_images >= self.num_visualizations:
            return

        images = batch["input"].detach().cpu().numpy()
        ids = batch["id"].detach().cpu().numpy()
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
            self.summary_writer.add_image(
                f'{tag}/detection_{ids[i]}', result, step)
            self.__num_logged_images += 1

            if self.__num_logged_images >= self.num_visualizations:
                break

    def log_stat(self, name, value, step):
        self.summary_writer.add_scalar(name, value, step)

    def reset(self):
        self.__num_logged_images = 0
