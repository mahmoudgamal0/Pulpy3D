import os
import time
import wandb
import logging
import logging.config
import torch
import torchio as tio
import torch.nn as nn
import numpy as np
import nibabel as nib

from tqdm import tqdm
from torch.utils.data import DataLoader

from losses.LossFactory import LossFactory
from models.ModelFactory import ModelFactory
from optimizers.OptimizerFactory import OptimizerFactory
from eval import Eval as Evaluator
from schedulers.SchedulerFactory import SchedulerFactory


class Experiment:
  def __init__(self, config, debug=False):
    self.config = config
    self.debug = debug
    self.epoch = 0
    self.eps = 1e-10
    self.metrics = {}

    self.num_classes = self.config.data_loader.num_classes

    # load model
    model_name = self.config.model.name
    in_ch = 1
    emb_shape = [dim // 8 for dim in self.config.data_loader.patch_shape]

    self.model = ModelFactory(model_name, self.num_classes, in_ch, emb_shape).get()
    if torch.cuda.is_available():
      self.model = self.model.cuda()

    self.model = nn.DataParallel(self.model)
    wandb.watch(self.model, log_freq=10)

    # load optimizer
    optim_name = self.config.optimizer.name
    train_params = self.model.parameters()
    lr = self.config.optimizer.learning_rate

    self.optimizer = OptimizerFactory(optim_name, train_params, lr).get()

    # load scheduler
    sched_name = self.config.lr_scheduler.name
    sched_milestones = self.config.lr_scheduler.get('milestones', None)
    sched_gamma = self.config.lr_scheduler.get('factor', None)
    sched_patience = self.config.lr_scheduler.get('patience', None)

    self.scheduler = SchedulerFactory(
      sched_name,
      self.optimizer,
      milestones=sched_milestones,
      gamma=sched_gamma,
      mode='max',
      verbose=True,
      patience=sched_patience
    ).get()

    # load loss
    self.loss = LossFactory(self.config.loss.name, classes=self.num_classes)

    # load evaluator
    self.evaluator = Evaluator(self.config, classes=self.num_classes)

    # Accumlator
    self.accumlation_iter = self.config.data_loader.accumlation_iter

  def save(self, name):
    if '.pth' not in name:
      name = name + '.pth'
    path = os.path.join(self.config.project_dir, self.config.title, 'checkpoints', name)
    logging.info(f'Saving checkpoint at {path}')
    state = {
      'title': self.config.title,
      'epoch': self.epoch,
      'state_dict': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'metrics': self.metrics,
    }
    torch.save(state, path)

  def load(self):
    path = self.config.trainer.checkpoint
    logging.info(f'Loading checkpoint from {path}')
    state = torch.load(path)

    if 'title' in state.keys():
      # check that the title headers (without the hash) is the same
      self_title_header = self.config.title[:-11]
      load_title_header = state['title'][:-11]
      if self_title_header == load_title_header:
        self.config.title = state['title']
    
    if state.get('optimizer', None) is not None:
      self.optimizer.load_state_dict(state['optimizer'])
    self.model.load_state_dict(state['state_dict'])
    self.epoch = state.get('epoch', 0) + 1

    if 'metrics' in state.keys():
      self.metrics = state['metrics']

  def extract_data_from_patch(self, patch):
    images = patch['data'][tio.DATA].float().cuda()
    gt = patch['gt'][tio.DATA].float().cuda()
    emb_codes = torch.cat((
      patch[tio.LOCATION][:,:3],
      patch[tio.LOCATION][:,:3] + torch.as_tensor(images.shape[-3:])
    ), dim=1).float().cuda()

    return images, gt, emb_codes

def predict(self):
  output_path = os.path.join(self.config.project_dir, self.config.title, 'outputs')
  timings_path = os.path.join(output_path, 'timings.csv')

  os.makedirs(output_path)
  with open(timings_path, 'w') as f:
    f.write("Sample,Time Taken\n")

  self.model.eval()
  with torch.inference_mode():
    dataset = self.test_dataset

    for _, subject in tqdm(enumerate(dataset), total=len(dataset), desc=f'Predict epoch {str(self.epoch)}'):
      sampler = tio.inference.GridSampler(
        subject,
        self.config.data_loader.patch_shape,
        0
      )
      loader = DataLoader(sampler, batch_size=self.config.data_loader.batch_size)
      aggregator = tio.inference.GridAggregator(sampler)
      start_time = time.time()
      for _, patch in enumerate(loader):
        images, emb_codes = self.extract_data_from_patch(patch)

        preds = self.model(images, emb_codes)
        aggregator.add_batch(preds, patch[tio.LOCATION])

      output = aggregator.get_output_tensor()
      end_time = time.time()
      output = output.squeeze(0)

      if self.num_classes > 1:
        output = torch.argmax(output, dim=0)
      else:
        output = (output > 0.5)

      with open(timings_path, 'a') as f:
        f.write(f"{subject.patient},{round(end_time-start_time,2)}\n")

      out = nib.Nifti1Image(output.cpu().numpy().astype(np.uint8), affine=np.eye(4), dtype=np.uint8)
      out.header.get_xyzt_units()
      out.to_filename(os.path.join(output_path, f'{subject.patient}.nii.gz'))
