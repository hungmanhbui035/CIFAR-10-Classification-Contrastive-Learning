import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import argparse
import os

from models import CNN, ResNet18, ViT
from train_test_utils import test

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--model-path', type=str, required=True)
    
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    classes = test_set.classes

    if args.model == 'cnn':
        model = CNN(num_classes=len(classes)).to(device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=len(classes)).to(device)
    elif args.model == 'vit':
        model = ViT(num_classes=len(classes)).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true, y_pred = test(model, loader, criterion, device)
    print(f"\ntest_loss {test_loss:.4f} test acc {test_acc:.2f}%")

    print(f"\n{classification_report(y_true, y_pred)}")

    if not os.path.exists('./cm'):
        os.makedirs('./cm')
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'./cm/{args.model}.png')
    plt.show()

if __name__ == '__main__':
    main()



