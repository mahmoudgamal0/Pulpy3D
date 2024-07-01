from experiments.segmentation import Segmentation
from experiments.semantic import Semantic
from experiments.instance import Instance
from experiments.multihead import MultiHead

class ExperimentFactory:
  def __init__(self, config, debug=False):
    self.name = config.experiment.name
    self.config = config
    self.debug = debug

  def get(self):
    if self.name == 'Segmentation':
      experiment = Segmentation(self.config, self.debug)
    elif self.name == 'Semantic':
      experiment = Semantic(self.config, self.debug)
    elif self.name == 'Instance':
      experiment = Instance(self.config, self.debug)
    elif self.name == 'MultiHead':
      experiment = MultiHead(self.config, self.debug)
    else:
      raise ValueError(f'Experiment \'{self.name}\' not found')
    return experiment
