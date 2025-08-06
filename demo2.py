import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import os

from utils.data_utils import get_misclassified_images, save_misclassified_images
from networks.cnn import CNN
from networks.resnet18 import ResNet18
from networks.vit import ViT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(mean=-mean/std, std=1/std)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    pil_transform = transforms.Compose([
        denormalize,
        transforms.ToPILImage(),
    ])

    if args.dataset == 'train':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tensor_transform)
    elif args.dataset == 'test':
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tensor_transform)
    
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    classes = len(dataset.classes)
    
    # model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    
    if args.model == 'cnn':
        model = CNN(out_dim=classes)
    elif args.model == 'resnet18':
        model = ResNet18(out_dim=classes)
    elif args.model == 'vit':
        model = ViT(out_dim=classes)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    misclassified_dict = get_misclassified_images(model, loader, pil_transform)
    save_misclassified_images(misclassified_dict, save_dir=f'./misclassified/{args.model}_{args.dataset}')

if __name__ == '__main__':
    main()