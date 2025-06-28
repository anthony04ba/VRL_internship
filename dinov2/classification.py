import torch
from torchvision import transforms
from PIL import Image
from dinov2.hub.classifiers import dinov2_vitb14_lc
import os


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return tensor

def pad_to_multiple(img_tensor, multiple=14):
    _, _, h, w = img_tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h))
    return img_tensor

if __name__ == "__main__":
    image_path = r
    classifier = dinov2_vitb14_lc(pretrained=True)
    classifier.cuda().eval()

    img_tensor = preprocess_image(image_path)
    img_tensor = pad_to_multiple(img_tensor, multiple=14)
    img_tensor = img_tensor.cuda()

    with torch.no_grad():
        logits = classifier(img_tensor)
        pred_class = logits.argmax(dim=1).item()
        

    with open('imagenet_classes.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace or newlines
            if line.startswith(f"{pred_class},"):
                print(line.split(",")[1])  # This is the line that starts with the given number
                break

