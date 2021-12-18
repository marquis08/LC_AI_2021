import segmentation_models_pytorch as smp
from torch import nn

# from hr_models.seg_hrnet_ocr import get_seg_model
import yaml
from munch import DefaultMunch
import torch.nn.functional as F

class SegModel(nn.Module):
    def __init__(self, encoder_name=None, arch_name=None):
        super().__init__() # SegModel, self

        if arch_name == "UnetPlusPlus":

            if encoder_name == 'tu-hrnet_w18':
                self.model = smp.UnetPlusPlus(
                    encoder_weights='imagenet',
                    encoder_name='tu-hrnet_w18', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'timm-efficientnet-b0':
                self.model = smp.UnetPlusPlus(
                    encoder_weights='imagenet',
                    encoder_name='timm-efficientnet-b0', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'timm-efficientnet-b3':
                self.model = smp.UnetPlusPlus(
                    encoder_weights='imagenet',
                    encoder_name='timm-efficientnet-b3', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'resnet18':
                self.model = smp.UnetPlusPlus(
                    encoder_weights='imagenet',
                    encoder_name='resnet18', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
        elif arch_name == "Unet":
            if encoder_name == 'tu-hrnet_w18':
                self.model = smp.Unet(
                    encoder_weights='imagenet',
                    encoder_name='tu-hrnet_w18', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'timm-efficientnet-b0':
                self.model = smp.Unet(
                    encoder_weights='imagenet',
                    encoder_name='timm-efficientnet-b0', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'timm-efficientnet-b3':
                self.model = smp.Unet(
                    encoder_weights='imagenet',
                    encoder_name='timm-efficientnet-b3', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )
            elif encoder_name == 'resnet18':
                self.model = smp.Unet(
                    encoder_weights='imagenet',
                    encoder_name='resnet18', #  tu-hrnet_w18 timm-efficientnet-b0
                    in_channels=3,
                    classes=8
                    )

    def forward(self, x):
        output = self.model(x)
        return output


# class HROCR(nn.Module):
#     def __init__(self):
#         super().__init__()

#         config_path='hr_ocr_2.yaml'
#         with open(config_path) as f:
#             cfg = yaml.load(f, Loader=yaml.FullLoader)

#         cfg = DefaultMunch.fromDict(cfg)
#         self.model = get_seg_model(cfg)
        
#     #@autocast()
#     def forward(self, x):
#         x = self.model(x)
#         x = F.interpolate(input=x[0], size=(512, 512), mode='bilinear', align_corners=True)
#         return x


if __name__ == '__main__':
    model = SegModel('timm-efficientnet-b0')