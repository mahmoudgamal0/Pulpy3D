import json
import torchio as tio


class SemanticLabelTransformation(tio.Transform):
  def __init__(self, ):
    super().__init__()

  def apply_transform(self, subject):
    label = subject['gt'][tio.DATA]
    label[label > 1] = 2
    subject['gt'][tio.DATA] = label
    return subject

class LabelTransformation(tio.Transform):
  def __init__(self):
    super().__init__()

  def apply_transform(self, subject):
    label = subject['gt'][tio.DATA]
    label[label > 1] = 1
    subject['gt'][tio.DATA] = label
    return subject
  
class InstanceLabelTransformation(tio.Transform):
  def __init__(self):
    super().__init__()
    self.mapping = json.load(open("./datasets/Pulpy3D/dataset.json"))['map']

  def apply_transform(self, subject):
    label = subject['gt'][tio.DATA]
    for current, target in self.mapping.items():
      label[label == int(current)] = int(target)
    subject['gt'][tio.DATA] = label
    return subject
  

class ShiftFromZero(tio.Transform):
  def __init__(self):
    super().__init__()

  def apply_transform(self, subject):
    data = subject.data[tio.DATA]
    min_value = data.min()
    if min_value < 0:
      data -= min_value
    
    subject.data[tio.DATA] = data
    return subject
  
class RemoveOutliers(tio.Transform):
  def __init__(self, std_multiplier=3):
    super().__init__()
    self.std_multiplier = std_multiplier

  def apply_transform(self, subject):
    data = subject.data[tio.DATA]

    mean = data.mean()
    std = data.std()
    lower_bound = mean - self.std_multiplier * std
    upper_bound = mean + self.std_multiplier * std
    data = data.clamp(lower_bound, upper_bound)
    
    subject.data[tio.DATA] = data
    return subject
  
class DynamicResample(tio.Resample):
  def __init__(self):
    super().__init__()

  def apply_transform(self, subject):
    self.target = round(max(subject.data.spacing), 1)
    return super().apply_transform(subject)
