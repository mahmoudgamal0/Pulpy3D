from torch import nn

class ModelFactory(nn.Module):
    def __init__(self, model_name, num_classes, in_ch, emb_shape=None):
        super(ModelFactory, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_ch = in_ch
        self.emb_shape = emb_shape

    def get(self):
        if self.model_name == 'AttentionPosPadUNet3D':
            from models.AttentionSegmentation.AttentionPosPadUNet3D import PosPadUNet3D
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'PosPadUNet3D':
            from models.PosPadUNet3D import PosPadUNet3D
            assert self.emb_shape is not None
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'UNETR':
            from models.UNETR import UNETR
            return UNETR(img_shape=(80,80,80), input_dim=self.in_ch, output_dim=self.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1)
        elif self.model_name == 'UNet3D':
            from models.UNet3D import UNet3d
            return UNet3d(in_channels=self.in_ch, out_channels=self.num_classes, init_features=16, image_shape=(80, 80, 80))
        elif self.model_name == 'VNet3D':
            from models.VNet3D import VNet3d
            return VNet3d(image_channel=self.in_ch, numclass=self.num_classes, init_features=16, image_shape=(80,80,80))
        elif self.model_name == 'AttentionInstancePosPadUNet3D':
            from models.AttentionSegmentation.AttentionInstancePosPadUNet3D import PosPadUNet3D
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)
        elif self.model_name == 'InstanceUNETR':
            from models.InstanceUNETR import UNETR
            return UNETR(img_shape=(80,80,80), input_dim=self.in_ch, output_dim=self.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1)
        elif self.model_name == 'FairyPosPadUNet3D':
            from models.IANSegmentation.PosPadUNet3D import PosPadUNet3D
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)   
        elif self.model_name == 'MultiHeadPosPadUNet3D':
            from models.MultiHead.AttentionPosPadUNet3D import PosPadUNet3D
            return PosPadUNet3D(self.num_classes, self.emb_shape, self.in_ch)       
        else:
            raise ValueError(f'Model {self.model_name} not found')
