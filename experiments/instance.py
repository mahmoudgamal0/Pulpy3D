import torchio as tio

from experiments.semantic import Semantic
from dataloader.Pulpy import Pulpy
from dataloader.Transformation import InstanceLabelTransformation

class Instance(Semantic):

  def __init__(self, config, debug=False):
    super().__init__(config, debug)

    self.train_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='train',
      labels={'gt': 'gt_instance_ian'},
      transform=tio.Compose([
        InstanceLabelTransformation(),
        tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
        self.config.data_loader.preprocessing,
        self.config.data_loader.augmentations,
      ]),
    )
    self.val_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='val',
      labels={'gt': 'gt_instance_ian'},
      transform=tio.Compose([
        InstanceLabelTransformation(),
        self.config.data_loader.preprocessing,
      ])
    )
    self.test_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='test',
      labels={'gt': 'gt_instance_ian'},
      transform=tio.Compose([
        InstanceLabelTransformation(),
        self.config.data_loader.preprocessing,
      ])
    )

    # queue start loading when used, not when instantiated
    self.train_loader = self.train_dataset.get_loader(self.config.data_loader)

    if self.config.trainer.reload:
      self.load()
