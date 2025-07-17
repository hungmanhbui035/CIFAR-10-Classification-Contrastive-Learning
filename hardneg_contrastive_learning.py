import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import os
import wandb

from data_utils import TransformedDataset, train_val_split, HardNegContrastiveDataset, MultiTransform, get_misclassified_images
from models import CNN, ResNet18, ViT
from contrastive_loss import HardNegContrastiveLoss
from train_test_utils import hcl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-positives', type=int, default=2)
    parser.add_argument('--num-hard-negatives', type=int, default=8)
    parser.add_argument('--num-randoms', type=int, default=54)
    parser.add_argument('--csf-batch-size', type=int, default=1024)
    parser.add_argument('--hcl-batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--cl-model-path', type=str, required=True)
    parser.add_argument('--sl-model-path', type=str, required=True)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--ckpt-path', type=str, default=None)

    parser.add_argument('--contrast-mode', type=str, default='scl')
    parser.add_argument('--temperature', type=int, default=0.1)
    
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=100)

    parser.add_argument('--log-freq', type=int, default=10)
    return parser.parse_args()


def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        tensor_transform
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    train_dataset, _ = train_val_split(dataset, val_ratio=0.2)
    csf_train_set = TransformedDataset(dataset=train_dataset, transform=tensor_transform)
    csf_loader = DataLoader(csf_train_set, batch_size=args.csf_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.model == 'cnn':
        csf_model = CNN(num_classes=10).to(device)
    elif args.model == 'resnet18':
        csf_model = ResNet18(num_classes=10).to(device)
    elif args.model == 'vit':
        csf_model = ViT(num_classes=10).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    csf_model.load_state_dict(torch.load(args.sl_model_path, map_location=device))
    csf_model = nn.DataParallel(csf_model)

    misclassified_dict = get_misclassified_images(csf_model, csf_loader, pil_transform)

    hardneg_train_set = HardNegContrastiveDataset(dataset=train_dataset,
                                                pos_transform=MultiTransform(transform=train_transform, num_transforms=args.num_positives), 
                                                hardneg_transform=train_transform, 
                                                hardneg_dict=misclassified_dict,
                                                num_hardnegs=args.num_hard_negatives,
                                                num_randoms=args.num_randoms,
                                                random_transform=train_transform)
    
    hardneg_loader = DataLoader(hardneg_train_set, batch_size=args.hcl_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/hcl_{args.model}.pth'

    if args.model == 'cnn':
        model = CNN(num_classes=args.projection_dim).to(device)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=args.projection_dim).to(device)
    elif args.model == 'vit':
        model = ViT(num_classes=args.projection_dim).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.load_state_dict(torch.load(args.cl_model_path, map_location=device))
    model = nn.DataParallel(model)
    
    criterion = HardNegContrastiveLoss(contrast_mode=args.contrast_mode, temperature=args.temperature)
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
    wandb.init(project='CIFAR-10-HardNeg-Contrastive-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = hcl_train(epoch, model, hardneg_loader, criterion, optimizer, scheduler, device, args.log_freq)
        cl_epoch_log(epoch, train_loss, args.num_epochs)
    
        if epoch % 50 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }
            ckpt_path = f'./ckpts/hcl_{args.model}_{epoch}.pth'
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