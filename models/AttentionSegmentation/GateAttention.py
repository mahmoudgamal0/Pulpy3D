import torch
from torch import nn
from torch.nn import functional as F
from models.AttentionSegmentation.networks_other import init_weights

class GatingSignal(nn.Module):
  def __init__(self, in_size, out_size, kernel_size=(1,1,1), stride=(1,1,1), padding=(1,1,1), groups=1, padding_mode='replicate'):
    super(GatingSignal, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv3d(in_size, out_size, kernel_size, stride, padding, groups=groups, padding_mode=padding_mode),
      nn.BatchNorm3d(out_size),
      nn.ReLU(inplace=True),
    )

    for m in self.children(): init_weights(m, init_type='kaiming')

  def forward(self, inputs):
    outputs = self.conv(inputs)
    return outputs

class AttentionBlock(nn.Module):
  def __init__(self, in_channels, gating_channels):
    super(AttentionBlock, self).__init__()

    self.in_channels = in_channels
    self.gating_channels = gating_channels
    self.inter_channels = in_channels // 2
    if self.inter_channels == 0:
      self.inter_channels = 1

    self.upsample_mode = 'trilinear'

    self.W = nn.Sequential(
      nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
      nn.BatchNorm3d(self.in_channels),
    )

    self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
    
    self.phi = nn.Conv3d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)

    self.psi = nn.Conv3d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    self.nl1 = lambda x: F.relu(x, inplace=True)

    # Initialise weights
    for m in self.children():
      init_weights(m, init_type='kaiming')

  def forward(self, x, g):
    '''
    :param x: (b, c, t, h, w)
    :param g: (b, g_d)
    :return:
    '''

    input_size = x.size()
    batch_size = input_size[0]
    assert batch_size == g.size(0)

    #############################
    # compute compatibility score

    # theta => (b, c, t, h, w) -> (b, i_c, t, h, w)
    theta_x = self.theta(x)
    # theta_x_size = theta_x.size()

    # phi   => (b, c, t, h, w) -> (b, i_c, t, h, w)
    phi_g = self.phi(g)

    # phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)

    # nl(theta.x + phi.g + bias) -> f = (b, i_c, t/s1, h/s2, w/s3)
    f = torch.add(theta_x, phi_g)
    f = F.relu(f)
    psi_f = self.psi(f)

    ############################################
    # normalisation -- scale compatibility score
    #  psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
    sigm_psi_f = torch.sigmoid(psi_f)

    # sigm_psi_f is attention map! upsample the attentions and multiply
    sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
    y = sigm_psi_f.expand_as(x) * x

    W_y = self.W(y)
    return W_y
