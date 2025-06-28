import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from dinov2.models.vision_transformer import vit_base #importing the model
import cv2

def load_model():
    model = vit_base()#vit base is a function that returns an instance of the model
    model.cuda().eval()#eval is a method in the nn.Module that dinovisiontransformer inherits from
    return model

def preprocess_image(image_path, resize_hw=(512, 960)):
    transform = transforms.Compose([
        transforms.Resize(resize_hw),#Resize the image to match the expected input size for the model
        transforms.ToTensor(),#Transform from (H,W,3) to (3,H,W) and normalizes it to (0,1)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],#Normalize the tensors according to the ImageNet dataset that the model was trained on
                             std=[0.229, 0.224, 0.225])#Mean and standard deviation for the 3 channels RGB
    ])
    image = Image.open(image_path).convert("RGB")#read the image and convert it to RGB in case not
    tensor = transform(image).cuda().unsqueeze(0) #.unsqueeze(0) adds 1 to the tensor which is the size of the batch
    #tensor becomes 4D, tensor = [1,3,H,W]
    #batch is the number of images inputted to the model at each forward pass
    return tensor

# Step 3: Extract patch features
def extract_features(model, img_tensor):
    with torch.no_grad():#no_grad means no gradient, in other terms no backpropagation only inference
        features = model.forward_features(img_tensor)#Forward features returns a dictionary
    patch_tokens = features["x_norm_patchtokens"]  # Shape: [1, N_patches=1920, 768]
    cls_token = features["x_norm_clstoken"]
    return patch_tokens,cls_token

# Step 4: PCA projection to RGB image
#patch size is 16x16 pixels
#patch tokens are the embedding vectors

#This function uses PCA to compress the size of the embedding vector from N to 3
def features_to_rgb(patch_tokens, image_hw=(512, 960), patch_size=16):
    B, N, D = patch_tokens.shape# B is the batch, N is the number of patches, D is the size of the embedding vectors
    H, W = image_hw[0] // patch_size, image_hw[1] // patch_size

    patch_map = patch_tokens[0].reshape(H * W, D).cpu().numpy()  # [H*W, 768]
    pca = PCA(n_components=3)
    patch_rgb = pca.fit_transform(patch_map)  # [H*W, 3]

    # Normalize to 0–1
    patch_rgb -= patch_rgb.min(0)
    patch_rgb /= patch_rgb.max(0)

    # Reshape to image
    feature_vis = patch_rgb.reshape(H, W, 3)
    return feature_vis

def visualize_feature_map(rgb_map):
    # Convert [0,1] to [0,255]
    rgb_uint8 = (rgb_map * 255).astype(np.uint8)

    # Show it as-is with PIL
    img = Image.fromarray(rgb_uint8)
    output_path = os.path.join(os.getcwd(), "output.png")
    img.save(output_path)

def visualize_feature_map_upsampled(rgb_map, output_size=(960,512), output_name="output.png"):
    rgb_uint8 = (rgb_map * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8)
    img = img.resize(output_size,cv2.INTER_NEAREST)  # If no interpolation specified, perform BICUBIC
    output_dir = os.path.join(os.getcwd(), "output_features")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    img.save(output_path)


if __name__ == "__main__":
    image_path = r"c:\Users\HP\Downloads\images.jpeg"  
    model = load_model()
    img_tensor = preprocess_image(image_path, resize_hw=(512, 960))
    patch_tokens, cls_token = extract_features(model, img_tensor)
    feature_map = features_to_rgb(patch_tokens, image_hw=(512, 960), patch_size=16)
    # Use the input image name for the output feature map
    input_filename = os.path.basename(image_path)
    output_name = f"feature_{os.path.splitext(input_filename)[0]}.png"
    visualize_feature_map_upsampled(feature_map, output_name=output_name)

