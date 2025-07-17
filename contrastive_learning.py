import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from data_utils import MultiTransform, train_val_split, TransformedDataset
from models import CNN, ResNet18, ViT
from contrastive_loss import ContrastiveLoss
from train_test_utils import cl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--ckpt-path', type=str, default=None)

    parser.add_argument('--contrast-mode', type=str, default='scl', choices=['scl', 'simclr'])
    parser.add_argument('--temperature', type=int, default=0.1)
    
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=1000)

    parser.add_argument('--log-freq', type=int, default=10)
    return parser.parse_args()

def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_set = TransformedDataset(train_dataset, MultiTransform(transform, num_transforms=args.num_views))

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if args.model == 'cnn':
        model = CNN(num_classes=args.projection_dim).to(device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.projection_dim).to(device)
    elif args.model == 'vit':
        model = ViT(num_classes=args.projection_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/cl_{args.model}.pth'
    
    criterion = ContrastiveLoss(contrast_mode=args.contrast_mode, temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
    else:
        start_epoch = 1

    wandb.login()
    wandb.init(project='CIFAR-10-Contrastive-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = cl_train(epoch, model, loader, criterion, optimizer, scheduler, device, args.log_freq)
        cl_epoch_log(epoch, train_loss, args.num_epochs)

        if epoch % 50 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            ckpt_path = f'./ckpts/cl_{args.model}_{epoch}.pth'
            torch.save(ckpt, ckpt_path)
            artifact = wandb.Artifact('ckpt', type='model')
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()