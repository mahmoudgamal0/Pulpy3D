import torch
import torchio as tio
import wandb

from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.experiment import Experiment
from dataloader.Pulpy import Pulpy
from dataloader.Transformation import LabelTransformation

class Segmentation(Experiment):

  def __init__(self, config, debug=False):
    super().__init__(config, debug)

    self.train_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='train',
      labels={'gt': 'gt_pulp'},
      transform=tio.Compose([
        LabelTransformation(),
        tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
        self.config.data_loader.preprocessing,
        self.config.data_loader.augmentations,
      ]),
    )
    self.val_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='val',
      labels={'gt': 'gt_pulp'},
      transform=tio.Compose([
        LabelTransformation(),
        self.config.data_loader.preprocessing,
      ])
    )
    self.test_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='test',
      labels={'gt': 'gt_pulp'},
      transform=tio.Compose([
        LabelTransformation(),
        self.config.data_loader.preprocessing,
      ])
    )

    # queue start loading when used, not when instantiated
    self.train_loader = self.train_dataset.get_loader(self.config.data_loader)

    if self.config.trainer.reload:
      self.load()

  def train(self):
    self.model.train()
    self.evaluator.reset_eval()

    losses = []
    self.optimizer.zero_grad()
    for i, d in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Train epoch {str(self.epoch)}'):
      images, gt, emb_codes = self.extract_data_from_patch(d)
      partition_weights = 1
      gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))

      if torch.sum(gt_count) == 0: continue
      partition_weights = (self.eps + gt_count) / torch.max(gt_count)

      preds = self.model(images, emb_codes)
      if self.num_classes == 1:
        assert preds.ndim == gt.ndim, f'Gt and output dimensions are not the same before loss. {preds.ndim} vs {gt.ndim}'
      
      loss = self.loss(preds, gt, partition_weights) 
      losses.append(loss.item() / self.accumlation_iter)
      loss.backward()

      if ((i + 1) % self.accumlation_iter == 0) or (i + 1 == len(self.train_loader)):
        self.optimizer.step()
        self.optimizer.zero_grad()
      
      preds = (preds > 0.5)
      self.evaluator.compute_metrics(preds, gt)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()
    self.evaluator.class_metric()

    self.metrics['Train'] = {
      'iou': epoch_iou,
      'dice': epoch_dice,
      'hd95': epoch_hd95,
    }

    wandb.log({
      f'Epoch': self.epoch,
      f'Train/Loss': epoch_train_loss,
      f'Train/Dice': epoch_dice,
      f'Train/IoU': epoch_iou,
      f'Train/HD95': epoch_hd95,
      f'Train/Lr': self.optimizer.param_groups[0]['lr']
    })

    return epoch_train_loss, epoch_iou, epoch_hd95


  def test(self, phase):
    self.model.eval()
    with torch.inference_mode():
      self.evaluator.reset_eval()
      losses = []

      if phase == 'Test':
        dataset = self.test_dataset
      elif phase == 'Validation':
        dataset = self.val_dataset

      for _, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'{phase} epoch {str(self.epoch)}'):
        sampler = tio.inference.GridSampler(
          subject,
          self.config.data_loader.patch_shape,
          0
        )
        loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
        aggregator = tio.inference.GridAggregator(sampler)
        gt_aggregator = tio.inference.GridAggregator(sampler)

        for _, patch in enumerate(loader):
          images, gt, emb_codes = self.extract_data_from_patch(patch)

          preds = self.model(images, emb_codes)
          aggregator.add_batch(preds, patch[tio.LOCATION])
          gt_aggregator.add_batch(gt, patch[tio.LOCATION])

        output = aggregator.get_output_tensor()
        gt = gt_aggregator.get_output_tensor()
        partition_weights = 1

        gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
          
        if torch.sum(gt_count) != 0:
          partition_weights = (self.eps + gt_count) / (self.eps + torch.max(gt_count))

        loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
        losses.append(loss.item())

        output = output.squeeze(0)
        output = (output > 0.5)

        self.evaluator.compute_metrics(output.unsqueeze(0), gt.unsqueeze(0))

      epoch_loss = sum(losses) / len(losses)
      epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()
      self.evaluator.class_metric()

      wandb.log({
          f'Epoch': self.epoch,
          f'{phase}/Loss': epoch_loss,
          f'{phase}/Dice': epoch_dice,
          f'{phase}/IoU': epoch_iou,
          f'{phase}/HD95': epoch_hd95
      })

      return epoch_iou, epoch_dice, epoch_hd95
