import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from utils.data_utils import MultiTransform, train_val_split, TransformedDataset
from utils.contrastive_loss import ContrastiveLoss
from utils.train_utils import cl_train, cl_epoch_log
from networks.cnn import CNN
from networks.resnet18 import ResNet18
from networks.vit import ViT

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--bs', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--proj-dim', type=int, default=128)

    parser.add_argument('--mode', type=str, default='scl', choices=['scl', 'simclr'])
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
    return parser.parse_args()

def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    train_dataset, _ = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, MultiTransform(transform, num_transforms=args.views))

    loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)

    # model
    if args.model == 'cnn':
        model = CNN(out_dim=args.proj_dim)
    elif args.model == 'resnet18':
        model = ResNet18(out_dim=args.proj_dim)
    elif args.model == 'vit':
        model = ViT(out_dim=args.proj_dim)

    model = model.to(device)
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/cl_{args.model}.pth'
    
    # train
    criterion = ContrastiveLoss(contrast_mode=args.mode)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb.login()
    wandb.init(project='CIFAR-10-Contrastive-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=10)

    for epoch in range(1, args.epochs + 1):
        train_loss = cl_train(model, loader, criterion, optimizer, scheduler, epoch, device)
        cl_epoch_log(train_loss, epoch, args.epochs)

    torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Training finished!')

if __name__ == '__main__':
    main()