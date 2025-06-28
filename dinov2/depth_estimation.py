import os
import math
import itertools
from functools import partial

import torch
import torch.nn.functional as F
from dinov2.eval.depth.models import build_depther

# Input and output paths
input_image = r"e:\Downloads_2\what-is-depth-of-field_orig.jpg"
output_folder = "output_depth"#output folder name
os.makedirs(output_folder, exist_ok=True)#make output folder

# Center padding class to handle ViT patch-size constraints
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple #Center padding multiple

    def _get_pad(self, size): #padding to handle ViT patch-size constraints
        new_size = math.ceil(size / self.multiple) * self.multiple #calculate the new size
        pad_size = new_size - size #calculate the padding size
        pad_size_left = pad_size // 2 #calculate the left padding size
        pad_size_right = pad_size - pad_size_left #calculate the right padding size
        return pad_size_left, pad_size_right

    @torch.inference_mode() #Function decorator to disable gradient computation and enable inference mode
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])) 
        return F.pad(x, pads) #pad the input tensor with the calculated padding

# Build depth estimation model
def create_depther(cfg, backbone_model, backbone_size, head_type): #Takes the config, backbone model, backbone size, and head type as input
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    #Extract features from the backbone model by using the get_intermediate_layers method 
    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):#makes sure the patch size is divisible by the multiple
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther

# Choose backbone which is the feature extractor
BACKBONE_SIZE = "base"  # options: small, base, large, giant
backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE] #backbone architecture based on size
backbone_name = f"dinov2_{backbone_arch}" #example dinov2_vitb14

#Backbone weights are imported from torch.hub in this case
# Load pretrained backbone from torch.hub
#hubconf.py file loads the model from github when torch.hub.load is called
#If the model is already downloaded, torch.hub will use the cached version, which is in torch_models in disk E
backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)  # Load the backbone model
backbone_model.eval().cuda()  # Move to GPU and set to evaluation mode

# Load depth head and config
import urllib
import mmcv
from mmcv.runner import load_checkpoint

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

HEAD_DATASET = "nyu"  # or "kitti" # it tells which dataset the head was trained on
HEAD_TYPE = "dpt"     # or "linear", "linear4" # it is the head type ex dpt for depth and linear for classification or segmentation

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py" #config
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth" #weights

cfg_str = load_config_from_url(head_config_url)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py") #loads the config file 


# Create full model and load weights
model = create_depther(cfg, backbone_model, BACKBONE_SIZE, HEAD_TYPE)#calls the create_depther function to create the model
load_checkpoint(model, head_checkpoint_url, map_location="cpu")#load the pre-trained weights to the model
model.eval().cuda()  # Move to GPU

# Load and prepare input image
from PIL import Image
import matplotlib
from torchvision import transforms

image = Image.open(input_image)
original_size = image.size  # Store original size (width, height)

# Remove resizing to prevent CUDA runtime errors
# (do not call resize_to_multiple, just use the image as-is)

def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3],  # remove alpha and scale
        transforms.Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
    ])

#Rendering means applying a colormap to the depth values
def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)[:, :, :3]
    return Image.fromarray(colors)

# Resize and process image

transform = make_depth_transform()
transformed_image = transform(image)
batch = transformed_image.unsqueeze(0).cuda()  # Move to GPU

# Warm-up 
# Run a few iterations to warm up the model
# This helps to ensure that the first inference is not affected by initial overhead
# Because first the model needs to initialize CUDA kernels and allocate memory which could take some time
for _ in range(5):
    with torch.no_grad(), torch.cuda.amp.autocast():
        _ = model.whole_inference(batch, img_meta=None, rescale=True)

# Measure inference time
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()
with torch.no_grad(), torch.cuda.amp.autocast():#Inference
    result = model.whole_inference(batch, img_meta=None, rescale=True)
ender.record()
torch.cuda.synchronize()# Wait for all kernels in the stream to finish
print(f"Inference time: {starter.elapsed_time(ender):.2f} ms")


# Render and save output depth map
depth_image = render_depth(result.squeeze().cpu())
# Resize depth image back to original input image size
if depth_image.size != original_size:
    depth_image = depth_image.resize(original_size, Image.BICUBIC)
input_filename = os.path.basename(input_image)
output_filename = f"depth_{input_filename}"
output_path = os.path.join(output_folder, output_filename)
depth_image.save(output_path)
print(f"Depth image saved to: {output_path}")
