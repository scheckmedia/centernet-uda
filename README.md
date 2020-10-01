# CenterNet with Unsupervised Domain Adaptation methods
This repository holds a small framework to evaluate unsupervised domain adaptation methods in combination with
a CenterNet object detection network.

## Implemented UDA Methods
- [ADVENT](https://arxiv.org/abs/1811.12833)
  - [x] Direct Entropy minimization
  - [ ] Minimizing entropy with adversarial learning (RP)
- [Domain Adaptation for Semantic Segmentation with Maximum Squares Loss](https://arxiv.org/abs/1909.13589)
  - [x] Max Squares minimization
  - [ ] Image-wise Class-balanced Weighting Factor (MT I)
  - [ ] Multi-level Self-produced guidance (MT I)
- [ ] [FDA](https://arxiv.org/abs/2004.05498) (MT I)

## Usage
This implementation uses [hydra](https://github.com/facebookresearch/hydra) as configuration framework.
Mainly you can check all available configuration options with:

    python train.py --help

### Training
To run training you can execute the following command which loads the default experiment (configs/defaults.yaml)

    python train.py

A custom experiment will load the default parameters, which you can easily overwrite by placing another yaml configuration file into the folder `configs/experiment/`. E.g. the experiment `entropy_minimization` can be executed with:

    python train.py experiment=entropy_minimization

This file overwrites the attribute `experiment` and the `uda` property.

### Configuration

```yaml
experiment: default # experiment name and also folder name (outputs/default) where logs a.s.o. are saved

# path to pretrained weights
# optimizer states are not restored
pretrained: /mnt/data/Projects/centernet-uda/weights/coco_dla_2x.pth

# path to last trained model
# will restore optimizer states
# if pretraned path is given, resume path is used
resume:

# defines the training model
model:
  # what kind of backend will be used
  # valid values are all available models in backends folder
  # currenlty only dla (Deep Layer Aggregation) is implemented
  backend:
    name: dla
    params: # valid params are listed in e.g. backends.dla.build
      num_layers: 34
      num_classes: 6
    # defines the loss function for centernet (should be as it is)
    # weights can be overwritten if necessary
    loss:
      name: centernet.DetectionLoss
      params:
        hm_weight: 1.0
        off_weight: 1.0
        wh_weight: 0.1
  # used unsupervised domain adaptation method for the experiment
  # in this case none UDA method is used, only the centernet is trained
  uda:

datasets:
  # paraemters for training dataset
  training:
    # all available readers are listed in datasets/
    name: datasets.coco
    # valid parameters are all paramters in __init__
    params:
      image_folder: /mnt/data/datasets/theodore_plus/images/
      annotation_file: /mnt/data/datasets/theodore_plus/coco/annotations/instances.json
      target_domain_glob:
        - /mnt/data/datasets/DST/2020-02-14-14h30m47s/*.jpg
        - /mnt/data/datasets/CEPDOF/**/*.jpg
      # data augmentation is implemented via imgaug
      # valid augmentors and parameters are listed here:
      # https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html
      augmentation:
        - GammaContrast:
            gamma: [0.2, 1.5]
        - Affine:
            translate_percent: [-0.1, 0.1]
            scale: [0.8, 1.3]
            rotate: [-45, 45]
        - AdditiveGaussianNoise:
            scale: [0, 10]
        - Fliplr:
            p: 0.5
        - Flipud:
            p: 0.5

  # paraemters for validation dataset
  validation:
    name: datasets.coco
    params:
      image_folder: /mnt/data/datasets/omnidetector-Flat/JPEGImages/
      annotation_file: /mnt/data/datasets/omnidetector-Flat/coco/annotations/instances_training.json
      target_domain_glob:
        - /mnt/data/datasets/DST/2020-02-14-14h30m47s/*.jpg
        - /mnt/data/datasets/CEPDOF/**/*.jpg

# parameters to normalize an image, additional to pixel / 255 normalization
normalize:
  mean: [0.40789654, 0.44719302, 0.47026115]
  std: [0.28863828, 0.27408164, 0.27809835]

# all available optimizer from https://pytorch.org/docs/stable/optim.html
# are valid values, e.g. using sgd instead of adam, name should be changed
# to SGD https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
# valid params are all listed params for an optimizer
# e.g. nesterov: True will be valid for SGD but not for Adam
optimizer:
  name: Adam # optimizer
  params:
    lr: 0.0001 # learning rate
  scheduler: # None or one of torch.optim.lr_scheduler
    name: MultiStepLR
    params:
      milestones: [20, 40]
      gamma: 0.1

# which evaluation framework should be used
# currently only mscoco is implemented but can be extended to pascal voc
evaluation:
  coco:
    per_class: True # if true for each class a mAP, Precision and Recall will be logged

tensorboard:
  num_visualizations: 50 # number of images with detections to show in tensorobard
  score_threshold: 0.3 # threshold which should be reached to be a valid bounding box

max_detections: 150 # maximum number of detections per image
epochs: 100 # number of epochs to train
batch_size: 16 # batch size
num_workers: 4 # number of parallel workers are used for the data loader

seed: 42 # random seed
gpu: 0 # gpu id to use for training


# how to identify a "best" model, what metric describes it
save_best_metric:
  name: validation/total_loss # can be training/total_loss, validation/total_loss or MSCOCO_Precision/mAP
  mode: min # what means best for the metric, is smaller (min) better or bigger (max)
```

# Development
WIP


### References
> [**Objects as Points**](http://arxiv.org/abs/1904.07850),
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*
