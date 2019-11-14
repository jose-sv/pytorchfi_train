'''Train, using pytorchfi instead as a fancy dropout'''
from __future__ import print_function
import argparse
import logging
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from pytorchfi import PyTorchFI_Util as pfi_util
from pytorchfi import PyTorchFI_Core as pfi_core
from resnet.models.resnet import ResNet18


def train(model, device, train_loader, optimizer):
    '''Train the model'''
    model.train()
    pbar = tqdm(train_loader, unit='Batches', desc='Training')
    criterion = nn.CrossEntropyLoss()
    # for batch_idx, (data, target) in enumerate(train_loader):
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f'{loss.item():.2f}')  # noqa


def get_lr(mizer):
    """Used for convenience when printing"""
    for param_group in mizer.param_groups:
        return param_group['lr']


def test(model, device, test_loader):
    '''Test the model on a validation set'''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()  # NOQA
    pbar = tqdm(test_loader, unit='Batches', desc='Testing')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target.to(device)).item()
            # get the index of the max log-probability
            # pred = output.argmax(dim=1, keepdim=True)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += len(data)
            pbar.set_postfix(correct=f'{correct/total * 100:.2f}%')

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    # logging.info('Validation Accuracy: %.4f Loss: %.4f', accuracy, test_loss)
    tqdm.write(
        f'Validation Accuracy: {accuracy:.4f}, Loss: {test_loss:.4f} '
        f'({correct}/{len(test_loader.dataset)})'
    )
    return accuracy


def main():
    '''Setup and iterate over training'''
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=450, metavar='N',
                        help='number of epochs to train (default: 450)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-pfi', action='store_true', default=False,
                        help='Use PFI as a dropout alternative')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-frequency', type=int, default=50)

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # pylint: disable=E1101
    device = torch.device("cuda" if use_cuda else "cpu")

    logging.info('Using %s', 'GPU' if use_cuda else 'CPU')

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(brightness=0.1,
                                                    contrast=0.1,
                                                    saturation=0.1),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/scratch/data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4,
                          momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=[150, 250, 350],
                            gamma=0.1)

    if args.use_pfi:
        mdl = model
        name = "pfi_cifar_resnet18.pt"
    else:
        mdl = model
        name = "cifar_resnet18.pt"

    acc = test(model, device, test_loader)
    with trange(1, args.epochs + 1, unit='Epoch', desc='Training') as pbar:
        for epoch in pbar:
            # validation every N epochs
            # also check to see if PFI should be turned on now
            if epoch % args.log_frequency == 0 and epoch != 0:
                acc = test(mdl, device, test_loader)

                # change to PFI at epoch 50
                # delayed to increase likelihood of convergence
                if args.use_pfi and epoch == 50:
                    pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
                    inj_model = pfi_util.random_inj_per_layer()
                    mdl = inj_model
                    pbar.set_desc('PFI Training')

            pbar.set_postfix(lr=get_lr(optimizer), acc=f'{acc:.2f}%')

            train(mdl, device, train_loader, optimizer)
            scheduler.step()

    if args.save_model:
        torch.save(mdl.state_dict(), name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
