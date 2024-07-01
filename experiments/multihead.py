import wandb
import torch

import torchio as tio

from tqdm import tqdm
from torch.utils.data import DataLoader

from eval import Eval as Evaluator
from experiments.experiment import Experiment
from experiments.experiment import Experiment
from dataloader.Pulpy import Pulpy


class MultiHead(Experiment):
  def __init__(self, config, debug=False):
    super().__init__(config, debug)
    self.pulp_evaluator = Evaluator(self.config, classes=self.num_classes)
    self.nerve_evaluator = Evaluator(self.config, classes=self.num_classes)

    self.train_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='train',
      labels={'gt_pulp': 'gt_pulp', 'gt_ian': 'gt_ian'},
      transform=tio.Compose([
        tio.CropOrPad(self.config.data_loader.resize_shape, padding_mode=0),
        self.config.data_loader.preprocessing,
        self.config.data_loader.augmentations,
      ]),
    )
    self.val_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='val',
      labels={'gt_pulp': 'gt_pulp', 'gt_ian': 'gt_ian'},
      transform=tio.Compose([
        self.config.data_loader.preprocessing,
      ])
    )
    self.test_dataset = Pulpy(
      root=self.config.data_loader.dataset,
      config=self.config.data_loader,
      splits='test',
      labels={'gt_pulp': 'gt_pulp', 'gt_ian': 'gt_ian'},
      transform=tio.Compose([
        self.config.data_loader.preprocessing,
      ])
    )

    # queue start loading when used, not when instantiated
    self.train_loader = self.train_dataset.get_loader(self.config.data_loader)

    if self.config.trainer.reload:
      self.load()


  def extract_data_from_patch(self, patch):
    images = patch['data'][tio.DATA].float().cuda()
    gt_pulp = patch['gt_pulp'][tio.DATA].float().cuda()
    gt_ian = patch['gt_ian'][tio.DATA].float().cuda()
    emb_codes = torch.cat((
      patch[tio.LOCATION][:,:3],
      patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
    ), dim=1).float().cuda()

    return images, gt_pulp, gt_ian, emb_codes

  def train(self):
    self.model.train()
    self.evaluator.reset_eval()

    data_loader = self.train_loader

    losses = []
    self.optimizer.zero_grad()
    for i, d in tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Train epoch {str(self.epoch)}'):
      images, gt_pulp, gt_ian, emb_codes = self.extract_data_from_patch(d)
      pulp_partition_weights = 1
      nerve_partition_weights = 1

      gt_pulp_count = torch.sum(gt_pulp == 1, dim=list(range(1, gt_pulp.ndim)))
      gt_ian_count = torch.sum(gt_ian == 1, dim=list(range(1, gt_ian.ndim)))

      if torch.sum(gt_pulp_count) + torch.sum(gt_ian_count) == 0: continue
      pulp_partition_weights = (self.eps + gt_pulp_count) / (torch.max(gt_pulp_count) + self.eps)
      nerve_partition_weights = (self.eps + gt_ian_count) / (torch.max(gt_ian_count) + self.eps)

      pulp_pred, nerve_pred = self.model(images, emb_codes)
      if self.num_classes == 1:
        assert pulp_pred.ndim == gt_pulp.ndim, f'Gt and output dimensions are not the same before loss. {pulp_pred.ndim} vs {gt_pulp.ndim}'
        assert nerve_pred.ndim == gt_ian.ndim, f'Gt and output dimensions are not the same before loss. {nerve_pred.ndim} vs {gt_ian.ndim}'
      
      pulp_loss = self.loss(pulp_pred, gt_pulp, pulp_partition_weights) 
      nerve_loss = self.loss(nerve_pred, gt_ian, nerve_partition_weights)

      loss = pulp_loss + nerve_loss
      losses.append(loss.item() / self.accumlation_iter)
      loss.backward()

      if ((i + 1) % self.accumlation_iter == 0) or (i + 1 == len(data_loader)):
        self.optimizer.step()
        self.optimizer.zero_grad()
      
      preds = pulp_pred + nerve_pred
      gt = gt_pulp + gt_ian

      if self.num_classes == 1:
        preds = (preds > 0.5)

      self.evaluator.compute_metrics(preds, gt)

    epoch_train_loss = sum(losses) / len(losses)
    epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()
    self.evaluator.mean_metric()

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

        pulp_aggregator = tio.inference.GridAggregator(sampler)
        gt_pulp_aggregator = tio.inference.GridAggregator(sampler)

        nerve_aggregator = tio.inference.GridAggregator(sampler)
        gt_ian_aggregator = tio.inference.GridAggregator(sampler)

        for _, patch in enumerate(loader):
          images, gt_pulp, gt_ian, emb_codes = self.extract_data_from_patch(patch)

          pulp_pred, nerve_pred = self.model(images, emb_codes)
          preds = pulp_pred + nerve_pred
          gt = gt_pulp + gt_ian

          aggregator.add_batch(preds, patch[tio.LOCATION])
          gt_aggregator.add_batch(gt, patch[tio.LOCATION])
          pulp_aggregator.add_batch(pulp_pred, patch[tio.LOCATION])
          gt_pulp_aggregator.add_batch(gt_pulp, patch[tio.LOCATION])
          nerve_aggregator.add_batch(nerve_pred, patch[tio.LOCATION])
          gt_ian_aggregator.add_batch(gt_ian, patch[tio.LOCATION])

        output = aggregator.get_output_tensor()
        gt = gt_aggregator.get_output_tensor()
        partition_weights = 1

        gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
          
        if torch.sum(gt_count) != 0:
          partition_weights = (self.eps + gt_count) / (self.eps + torch.max(gt_count))

        loss = self.loss(output.unsqueeze(0), gt.unsqueeze(0), partition_weights)
        losses.append(loss.item())

        pulp_output = pulp_aggregator.get_output_tensor().squeeze(0)
        gt_pulp_output = gt_pulp_aggregator.get_output_tensor().squeeze(0)
        
        nerve_output = nerve_aggregator.get_output_tensor().squeeze(0)
        gt_ian_output = gt_ian_aggregator.get_output_tensor().squeeze(0)
        
        self.pulp_evaluator.compute_metrics((pulp_output > 0.5).unsqueeze(0), gt_pulp_output.unsqueeze(0))
        self.nerve_evaluator.compute_metrics((nerve_output > 0.5).unsqueeze(0), gt_ian_output.unsqueeze(0))

        output = pulp_output + nerve_output
        self.evaluator.compute_metrics((output > 0.5).unsqueeze(0), gt.unsqueeze(0))

      epoch_loss = sum(losses) / len(losses)
      epoch_iou, epoch_dice, epoch_hd95 = self.evaluator.mean_metric()
      pulp_iou, pulp_dice, pulp_hd95 = self.pulp_evaluator.mean_metric()
      nerve_iou, nerve_dice, nerve_hd95 = self.nerve_evaluator.mean_metric()
      
      wandb.log({
          f'Epoch': self.epoch,
          f'{phase}/Loss': epoch_loss,
          f'{phase}/Dice': epoch_dice,
          f'{phase}/IoU': epoch_iou,
          f'{phase}/HD95': epoch_hd95,
          f'{phase}/PULP_Dice': pulp_dice,
          f'{phase}/PULP_IoU': pulp_iou,
          f'{phase}/PULP_HD95': pulp_hd95,
          f'{phase}/NERVE_Dice': nerve_dice,
          f'{phase}/NERVE_IoU': nerve_iou,
          f'{phase}/NERVE_HD95': nerve_hd95
      })

      return epoch_iou, epoch_dice, epoch_hd95
