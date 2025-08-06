import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from utils.data_utils import TransformedDataset, train_val_split
from utils.train_utils import EarlyStopper, sl_train, sl_validate, sl_epoch_log
from networks.cnn import CNN
from networks.resnet18 import ResNet18
from networks.vit import ViT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--proj-dim', type=int, default=128)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early-stop', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        val_transform
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    train_dataset, val_dataset = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, train_transform)
    val_set = TransformedDataset(val_dataset, val_transform)

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    classes = len(dataset.classes)

    # model
    if not os.path.exists(args.model_path):
        raise ValueError(f'Path does not exist: {args.model_path}')
    
    if args.model == 'cnn':
        model = CNN(out_dim=args.proj_dim)
    elif args.model == 'resnet18':
        model = ResNet18(out_dim=args.proj_dim)
    elif args.model == 'vit':
        model = ViT(out_dim=args.proj_dim)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.fc = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, classes)
    )
    model = model.to(device)
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/sl_cl_{args.model}.pth'
    
    # train
    for param in model.parameters():
        if param not in model.module.fc.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(model.module.fc.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if args.early_stop:
        early_stopper = EarlyStopper(model, model_path, patience=5, min_delta=0.3)

    wandb.login()
    wandb.init(project='CIFAR-10-Classification')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=10)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = sl_train(model, train_loader, criterion, optimizer, scheduler, epoch, device)
        val_loss, val_acc = sl_validate(model, val_loader, criterion, epoch, device)
        sl_epoch_log(train_loss, train_acc, val_loss, val_acc, epoch, args.epochs)

        if args.early_stop:
            if early_stopper.early_stop(val_loss):
                print(f'Early stop at epoch {epoch}!')
                break

    if not args.early_stop:
        torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Training finished!')

if __name__ == '__main__':
    main()