import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from data_utils import TransformedDataset, train_val_split
from models import CNN, ResNet18, ViT
from train_test_utils import EarlyStopper, sl_train, sl_validate, sl_epoch_log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--ckpt-path', type=str, default=None)
    
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=0.3)

    parser.add_argument('--log-freq', type=int, default=10)
    return parser.parse_args()


def main():
    # parse args
    args = parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        val_transform
    ])

    # train dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    # train/val split
    train_dataset, val_dataset = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, train_transform)
    val_set = TransformedDataset(val_dataset, val_transform)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    if not args.model_path:
        raise ValueError('model_path is required')
    if args.model == 'cnn':
        model = CNN(num_classes=args.projection_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.projection_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.fc = nn.Linear(512, 10).to(device)
    elif args.model == 'vit':
        model = ViT(num_classes=args.projection_dim).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.fc = nn.Linear(256, 10).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/sl_{args.model}.pth'
    
    # optimizer, criterion, scheduler and early_stopper
    for param in model.parameters():
        param.requires_grad = False
    for param in model.module.fc.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.module.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    if args.cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None
    if args.early_stop:
        early_stopper = EarlyStopper(model, model_path, patience=args.patience, min_delta=args.min_delta)
    else:
        early_stopper = None

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 1

    # wandb
    wandb.login()
    wandb.init(project='CIFAR-10-Supervised-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    # train
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss, train_acc = sl_train(epoch, model, train_loader, criterion, optimizer, scheduler, device, args.log_freq)
        val_loss, val_acc = sl_validate(epoch, model, val_loader, criterion, device)
        sl_epoch_log(epoch, train_loss, train_acc, val_loss, val_acc, args.num_epochs)

        if early_stopper and early_stopper.early_stop(val_loss):
            print('Early stop at epoch', epoch)
            break
        
        if epoch % 50 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            ckpt_path = f'./ckpts/sl_{args.model}_{epoch}.pth'
            torch.save(ckpt, ckpt_path)
            artifact = wandb.Artifact('ckpt', type='model')
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    # log model to wandb
    if not early_stopper:
        torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()