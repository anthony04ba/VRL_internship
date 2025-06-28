import os
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from mmseg.apis import MMSegInferencer  # Updated import
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import numpy as np
from PIL import Image
import urllib.request
import dinov2.eval.segmentation.utils.colormaps as colormaps

# Center padding class to handle ViT patch-size constraints
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)

# Build segmentation model - Updated for modern API
def create_segmenter(cfg, backbone_model):
    # Convert config to dictionary if needed
    if hasattr(cfg, '_cfg_dict'):
        cfg_dict = cfg._cfg_dict
    else:
        cfg_dict = cfg
    
    # Create model using modern API
    model = MMSegInferencer(cfg_dict, checkpoint=None, device='cuda:0')
    
    # Modify backbone
    model.model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    
    if hasattr(backbone_model, "patch_size"):
        model.model.backbone.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
        )
    
    return model

# Choose backbone
BACKBONE_SIZE = "small"
backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"

# Load pretrained DINOv2 backbone
backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
backbone_model.eval().cuda()

# Load segmentation head config
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

HEAD_SCALE_COUNT = 3
HEAD_DATASET = "voc2012"
HEAD_TYPE = "ms"

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

cfg_str = load_config_from_url(head_config_url)
cfg = Config.fromstring(cfg_str, file_format=".py")

if HEAD_TYPE == "ms":
    cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1]["img_ratios"][:HEAD_SCALE_COUNT]
    print("scales:", cfg.data.test.pipeline[1]["img_ratios"])

# Build model and load weights
model = create_segmenter(cfg, backbone_model=backbone_model)
load_checkpoint(model.model, head_checkpoint_url, map_location="cpu")  # Updated to access .model
model.model.cuda().eval()  # Updated to access .model

# Load example image
def load_image_from_url(url: str) -> Image.Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")

EXAMPLE_IMAGE_URL = "https://dl.fbaipublicfiles.com/dinov2/images/example.jpg"
image = load_image_from_url(EXAMPLE_IMAGE_URL)

# Segmentation and visualization
DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)

# Perform inference using the modern API
array = np.array(image)[:, :, ::-1]  # Convert RGB to BGR
result = model(array)  # Using modern inference API
segmentation_logits = result[0]  # Get first result

segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)
segmented_image.show()