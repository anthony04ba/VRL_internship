"""
Export the torch hub model to ONNX format. Normalization is done in the model.
"""

import torch
import fire

class Metric3DExportModel(torch.nn.Module):
    """
    The model for exporting to ONNX format. Add custom preprocessing and postprocessing here.
    """

    def __init__(self, meta_arch):#init method of the Metric3DExportModel class
        super().__init__()
        #meta_arch is the actual model
        self.meta_arch = meta_arch
        self.register_buffer(
            #These are normalization parameters (the same as the imageNet values)
            "rgb_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).cuda()#mean 
        )
        self.register_buffer(
            "rgb_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).cuda()#standard deviation
        )
        self.input_size = (616, 1064)

    def normalize_image(self, image):#method that normalizes the input image (R,G,B)
        image = image - self.rgb_mean
        image = image / self.rgb_std
        return image

    def forward(self, image):#method to predict depth, calls inference method
        image = self.normalize_image(image)#normalize the image by calling the normalize_image method
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.meta_arch.inference(
                {"input": image}
            )
        return pred_depth

#In case using a vit model use this method and if using kitty model do not use it
def update_vit_sampling(model):#normal function
    """
    For ViT models running on some TensorRT version, we need to change the interpolation method from bicubic to bilinear.
    """
    import torch.nn as nn
    import math

    def interpolate_pos_encoding_bilinear(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bilinear",  # Change from bicubic to bilinear
            antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    model.depth_model.encoder.interpolate_pos_encoding = (
        interpolate_pos_encoding_bilinear.__get__(
            model.depth_model.encoder, model.depth_model.encoder.__class__
        )
    )
    return model

#exporting the model to onnx function

def main(model_name="metric3d_hourglass", modify_upsample=False):
    from mmengine.config import Config
    from mono.model.monodepth_model import get_configured_monodepth_model

    # Replace with your actual config path
    if "vit" in model_name:#if the model is a vit, then import the vit_small model
        CONFIG_PATH = "mono/configs/VITDecoder/metric3d_vit_small.py"
    else:#if the model is an hourglassdecoder, then import the test_kitty_convlarge model
        CONFIG_PATH = "mono/configs/HourglassDecoder/test_kitti_convlarge.0.3_150.py"

    cfg = Config.fromfile(CONFIG_PATH)
    model = get_configured_monodepth_model(cfg).cuda().eval()

    if modify_upsample:#only for vit models
        model = update_vit_sampling(model)

    # Dummy input
    B = 1
    if "vit" in model_name:
        dummy_image = torch.randn([B, 3, 616, 1064]).cuda()
    else:
        dummy_image = torch.randn([B, 3, 512, 960]).cuda()#it creates a dummy input tensor for tracing


    export_model = Metric3DExportModel(model).cuda().eval()

    # Export to ONNX
    onnx_output = f"{model_name}.onnx"
    dummy_input = (dummy_image,)
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_output,
        input_names=["image"],
        output_names=["pred_depth"],
        opset_version=11, #which ONNX version the model uses
    )

    print(f"Succesfully Exported ONNX model to {onnx_output}")


if __name__ == "__main__":#if the code was ran separately, execute
    #if the code was imported by another function or class do not execute this
    from fire import Fire #fire is a python CLI or command line interpreter
    #It turns python functions or classes into commands that can be executed by the shell's OS
    #It lets you call the main function via the terminal
    Fire(main)
