import pytorch_lightning as pl
import torch

from .EfficientMedNext import create_efficient_mednext
from .emmamba import emmamba
from .nnunet import NNUNet3D
from .segformer3d import SegFormer3D
from .segmamba import SegMamba
from .swinunet import SwinUnet
from .unet_3d import UNet3D, UNet3DSmall


def _load_pl_checkpoint(model, weights):
    print(f"Loading a pretrained encoder-decoder from {weights} (a pytorch_lightning checkpoint)")
    state_dict = torch.load(weights, map_location='cpu')

    temp = TempModule(model)
    temp.load_state_dict(state_dict['state_dict'])

    return temp.get_model()


def get_model(
        encoder_name,
        weights=None,
        pretrained_type='encoder',
        img_size=None,
):
    encoder_name = encoder_name.lower()

    if encoder_name == '3d_unet':
        model = UNet3D()
        if (weights is not None) and (pretrained_type.lower() == 'pl_full'):
            model = _load_pl_checkpoint(model, weights)

    elif encoder_name == '3d_unet_small':
        model = UNet3DSmall()
        if (weights is not None) and (pretrained_type.lower() == 'pl_full'):
            model = _load_pl_checkpoint(model, weights)

    elif encoder_name == 'segmamba':
        model = SegMamba(in_chans=3, out_chans=1)

    elif encoder_name == 'emmamba':
        model = emmamba(
            in_chans=3,
            out_chans=1,
            depths=[2, 2, 2, 2],
            feat_size=[16, 32, 64, 128],
        )

    elif encoder_name == 'swinunet':
        model = SwinUnet(
            img_size=img_size or (128, 128, 32),
            in_channels=3,
            num_classes=1,
        )

    elif encoder_name == 'segformer3d':
        model = SegFormer3D()

    elif encoder_name == 'nnunet':
        model = NNUNet3D(in_channels=3, out_channels=1)

    elif encoder_name == 'efficientmednext':
        model = create_efficient_mednext(
            num_input_channels=3,
            num_classes=1,
            model_id='L',
            n_channels=32,
            deep_supervision=False,
            mode='train',
        )

    else:
        raise NotImplementedError(f"{encoder_name} is not recognized ...")

    return model


class TempModule(pl.LightningModule):
    def __init__(
            self,
            model,
    ):
        super().__init__()

        self.model = model

    def get_model(self):
        return self.model
