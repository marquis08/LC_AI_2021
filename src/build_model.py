import segmentation_models_pytorch as smp
from torch import nn

def build_model(args):
    model = SegModel(args)
    return model

class SegModel(nn.Module):
    def __init__(self, args):
        super(SegModel, self).__init__()

        if args.arch == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name=args.encoder_name,
                encoder_weights=args.encoder_weights,
                in_channels=3,
                classes=args.num_classes
                )
        elif args.arch == "Unet":
            self.model = smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights=args.encoder_weights,
                in_channels=3,
                classes=args.num_classes
                )
        elif args.arch == "DeepLabV3":
            self.model = smp.DeepLabV3(
                encoder_name=args.encoder_name,
                encoder_weights=args.encoder_weights,
                in_channels=3,
                classes=args.num_classes
                )

    def forward(self, x):
        output = self.model(x)
        return output
