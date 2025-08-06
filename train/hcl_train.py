import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import os
import wandb

from utils.data_utils import TransformedDataset, train_val_split, HardNegContrastiveDataset, MultiTransform, get_misclassified_images
from utils.contrastive_loss import HardNegContrastiveLoss
from utils.train_utils import hcl_train, cl_epoch_log
from networks.cnn import CNN
from networks.resnet18 import ResNet18
from networks.vit import ViT


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csf-bs', type=int, default=1024)
    parser.add_argument('--hcl-bs', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet18', 'vit'])
    parser.add_argument('--cl-model-path', type=str, required=True)
    parser.add_argument('--sl-model-path', type=str, required=True)
    parser.add_argument('--proj-dim', type=int, default=128)

    parser.add_argument('--mode', type=str, default='scl')
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()


def main():
    args = arg_parser()

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
    csf_loader = DataLoader(csf_train_set, batch_size=args.csf_bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    classes = len(dataset.classes)

    if args.model == 'cnn':
        csf_model = CNN(out_dim=classes)
    elif args.model == 'resnet18':
        csf_model = ResNet18(out_dim=classes)
    elif args.model == 'vit':
        csf_model = ViT(out_dim=classes)
    csf_model = csf_model.to(device)
    csf_model.load_state_dict(torch.load(args.sl_model_path, map_location=device))
    csf_model = nn.DataParallel(csf_model)

    misclassified_dict = get_misclassified_images(csf_model, csf_loader, pil_transform)

    hardneg_train_set = HardNegContrastiveDataset(dataset=train_dataset,
                                                  pos_transform=MultiTransform(transform=train_transform, num_transforms=2), 
                                                  hardneg_transform=train_transform, 
                                                  num_hardnegs=8,
                                                  random_transform=train_transform,
                                                  num_randoms=54,
                                                  hardneg_dict=misclassified_dict)
    
    hardneg_loader = DataLoader(hardneg_train_set, batch_size=args.hcl_bs, shuffle=True, num_workers=args.workers, pin_memory=True)

    # model
    if args.model == 'cnn':
        model = CNN(out_dim=args.proj_dim)
    elif args.model == 'resnet18':
        model = ResNet18(out_dim=args.proj_dim)
    elif args.model == 'vit':
        model = ViT(out_dim=args.proj_dim)
    model = model.to(device)
    model.load_state_dict(torch.load(args.cl_model_path, map_location=device))
    model = nn.DataParallel(model)

    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_path = f'./models/hcl_sl_cl_{args.model}.pth'

    # train
    criterion = HardNegContrastiveLoss(contrast_mode=args.mode)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    wandb.login()
    wandb.init(project='CIFAR-10-HardNeg-Contrastive-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=10)

    for epoch in range(1, args.epochs + 1):
        train_loss = hcl_train(model, hardneg_loader, criterion, optimizer, scheduler, epoch, device)
        cl_epoch_log(train_loss, epoch, args.epochs)

    torch.save(model.module.state_dict(), model_path)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print('Training finished!')

if __name__ == '__main__':
    main()