import os
import cv2
import torch
import numpy as np
from mmengine.config import Config
from mono.model.monodepth_model import get_configured_monodepth_model
from torchvision import transforms


CONFIG_PATH = 'mono/configs/HourglassDecoder/test_kitti_convlarge.0.3_150.py'#This model was trained and tested on the kitti dataset

IMAGE_PATH = r'data/kitti_demo/rgb/0000000005.png'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


cfg = Config.fromfile(CONFIG_PATH)
model = get_configured_monodepth_model(cfg).cuda().eval()


img = cv2.imread(IMAGE_PATH)
image_shape = (img.shape[1],img.shape[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#converts from BGR to RGB as expected by the model
img = cv2.resize(img, (960, 512))#resize the image to the model's expected input


transform = transforms.Compose([
    transforms.ToTensor(),#converts from (H,W,3) to (3,H,W) and normalizes to (0,1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #Normalizes each pixel according to ImageNet training
                         std=[0.229, 0.224, 0.225])# mean and standard deviation for each channel
])


img_tensor = transform(img).unsqueeze(0).cuda()#.cuda gives the operation to the GPU
#.unsqueeze adds a 1 at the beginning (1,3,H,W) where 1 is just the batch size

#The model and the input must be on the same device (Ex:GPU)
#So that's why .cuda() is present at the model instantiation
#And also we place the image tensor input on the GPU too!
#The model expects the input data as a dict


input_data = {'input': img_tensor}

pred_depth, confidence, output_dict = model.inference(input_data)



depth_np = pred_depth.squeeze().cpu().numpy() #.squeeze removes the batch size (squeezes the output)
#Now depth_np is a 2D numpy matrix (H,W), we switch to numpy because cv2 post processing requires arrays in numpy format
#The output is given back to the cpu because numpy only operates on CPU

depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)#Normalizes each pixel back again
#Avoid division by zero if max and min depth were equal
depth_img = (depth_norm * 255).astype(np.uint8)
depth_img = cv2.resize(depth_img,image_shape)
#now scale the values from 0-1 to 0-255 with each pixel size of 8 bits
#now for the colormap, each range of numbers from 0 to 255 represents a color
depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO,cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'depth.png'), depth_colored)


conf_tensor = output_dict['confidence'].squeeze().cpu().numpy()
conf_norm = (conf_tensor - conf_tensor.min()) / (conf_tensor.max() - conf_tensor.min() + 1e-8)
conf_img = (conf_norm * 255).astype(np.uint8)
conf_img = cv2.resize(conf_img,image_shape)
conf_colored = cv2.applyColorMap(conf_img, cv2.COLORMAP_RAINBOW,cv2.INTER_LANCZOS4)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'confidence.png'), conf_colored)



