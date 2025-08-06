import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import argparse
import os
from PIL import Image

from networks.cnn import CNN
from networks.resnet18 import ResNet18
from networks.vit import ViT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--image-path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found at {args.image_path}")
    
    image = Image.open(args.image_path)
    image_tensor = transform(image).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    if args.model == 'cnn':
        model = CNN(out_dim=len(classes))
    elif args.model == 'resnet18':
        model = ResNet18(out_dim=len(classes))
    elif args.model == 'vit':
        model = ViT(out_dim=len(classes))
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(output.data, 1)

    predicted_class = classes[predicted.item()]
    print(f"\nPredicted class: {predicted_class}")
    print("\nClass probabilities:")
    for i, class_name in enumerate(classes):
        probability = probabilities[0][i].item() * 100
        print(f"{class_name:10s}: {probability:.2f}%")

if __name__ == "__main__":
    main()