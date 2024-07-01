import torch
import torch.nn as nn
from models.AttentionSegmentation.GateAttention import (
    GatingSignal,
    AttentionBlock
)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class PosPadUNet3D(nn.Module):
    def __init__(self, n_classes, emb_shape, in_ch, size=32):
        self.n_classes = n_classes
        self.in_ch = in_ch
        super(PosPadUNet3D, self).__init__()

        self.emb_shape = torch.as_tensor(emb_shape)
        self.pos_emb_layer = nn.Linear(6, torch.prod(self.emb_shape).item())
        self.ec0 = self.conv3Dblock(self.in_ch, size, groups=1)
        self.ec1 = self.conv3Dblock(size, size*2, kernel_size=3, padding=1, groups=1)  # third dimension to even val
        self.ec2 = self.conv3Dblock(size*2, size*2, groups=1)
        self.ec3 = self.conv3Dblock(size*2, size*4, groups=1)
        self.ec4 = self.conv3Dblock(size*4, size*4, groups=1)
        self.ec5 = self.conv3Dblock(size*4, size*8, groups=1)
        self.ec6 = self.conv3Dblock(size*8, size*8, groups=1)
        self.ec7 = self.conv3Dblock(size*8, size*16, groups=1)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = nn.ConvTranspose3d(size*16+1, size*16, kernel_size=2, stride=2)
        self.dc8 = self.conv3Dblock(size*8 + size*16, size*8, kernel_size=3, stride=1, padding=1)
        self.dc7 = self.conv3Dblock(size*8, size*8, kernel_size=3, stride=1, padding=1)
        self.dc6 = nn.ConvTranspose3d(size*8, size*8, kernel_size=2, stride=2)
        self.dc5 = self.conv3Dblock(size*4 + size*8, size*4, kernel_size=3, stride=1, padding=1)
        self.dc4 = self.conv3Dblock(size*4, size*4, kernel_size=3, stride=1, padding=1)
        self.dc3 = nn.ConvTranspose3d(size*4, size*4, kernel_size=2, stride=2)
        self.dc2 = self.conv3Dblock(size*2 + size*4, size*2, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.conv3Dblock(size*2, size*2, kernel_size=3, stride=1, padding=1)

        self.final = nn.ConvTranspose3d(size*2, n_classes, kernel_size=3, padding=1, stride=1)
        
        ################
        # Attention blocks

        self.gating9 = GatingSignal(size*16+1, size*16, kernel_size=3, stride=1)
        self.gating6 = GatingSignal(size*8, size*8, kernel_size=3, stride=1)
        self.gating3 = GatingSignal(size*4, size*4, kernel_size=3, stride=1)

        self.attentionblock9 = AttentionBlock(in_channels=size*16+1, gating_channels=size*16)
        self.attentionblock6 = AttentionBlock(in_channels=size*8, gating_channels=size*8)
        self.attentionblock3 = AttentionBlock(in_channels=size*4, gating_channels=size*4)
        #################

        initialize_weights(self)

    def conv3Dblock(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), groups=1, padding_mode='replicate'):
        return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, groups=groups),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
        )

    def forward(self, x, emb_codes):
        hid = self.ec0(x)
        feat_0 = self.ec1(hid)
        hid = self.pool0(feat_0)
        hid = self.ec2(hid)
        feat_1 = self.ec3(hid)

        hid = self.pool1(feat_1)
        hid = self.ec4(hid)
        feat_2 = self.ec5(hid)

        hid = self.pool2(feat_2)
        hid = self.ec6(hid)
        hid = self.ec7(hid)

        emb_pos = self.pos_emb_layer(emb_codes).view(-1, 1, *self.emb_shape)
        hid = torch.cat((hid, emb_pos), dim=1)
        
        # Attention Mechanism
        gating9 = self.gating9(hid)
        g_dc9 = self.attentionblock9(hid, gating9)
        hid = torch.cat((self.dc9(g_dc9), feat_2), dim=1)

        hid = self.dc8(hid)
        hid = self.dc7(hid)

        gating6 = self.gating6(hid)
        g_dc6 = self.attentionblock6(hid, gating6)
        hid = torch.cat((self.dc6(g_dc6), feat_1), dim=1)
        hid = self.dc5(hid)
        hid = self.dc4(hid)

        gating3 = self.gating3(hid)
        g_dc3 = self.attentionblock3(hid, gating3)
        hid = torch.cat((self.dc3(g_dc3), feat_0), dim=1)
        hid = self.dc2(hid)
        hid = self.dc1(hid)
        
        hid = self.final(hid)
        return torch.sigmoid(hid)

if __name__ == '__main__':
    model = PosPadUNet3D(1, [10,10,10], 1)
    pos = torch.rand((1,6))
    x = torch.rand((1, 1, 80, 80, 80))
    model(x, pos)

