'''Train, using pytorchfi instead as a fancy dropout'''
from __future__ import print_function
import argparse
import logging
import pdb
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from pytorchfi import PyTorchFI_Util as pfi_util
from pytorchfi import PyTorchFI_Core as pfi_core
from resnet.models.resnet import ResNet18


# early termination by signal
TERMINATE = False


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
    return accuracy, test_loss, correct, len(test_loader.dataset)


def eval_confidences(model, device, test_loader):
    '''Calculate statistics on the confidences across a dataset'''
    return -1.0, np.zeros(10)
    model.eval()
    correct = 0
    confidences = np.zeros(10)
    pbar = tqdm(test_loader, unit='Batches', desc='Confidences')
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = F.softmax(model(data), dim=1)
            pdb.set_trace()
            confidences += output.sum(dim=0).cpu().numpy() / 128
            # get the index of the max log-probability
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    confidences /= len(test_loader.dataset)
    return accuracy, confidences


def main():
    '''Setup and iterate over training'''
    global TERMINATE

    use_cuda = not args.no_cuda and torch.cuda.is_available()

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

    scheduler = MultiStepLR(optimizer, milestones=[150, 250],
                            gamma=0.1)

    if args.use_pfi:
        mdl = model
        name = "pfi_cifar_resnet18.pt"
        logging.info('Using PFI from epoch %i', args.pfi_epoch)
    else:
        mdl = model
        name = "cifar_resnet18.pt"

    acc, loss, cor, tot = test(model, device, test_loader)
    with trange(1, args.epochs + 1, unit='Epoch', desc='Training') as pbar:
        tqdm.write(
            f'Validation Accuracy (-1): {acc:.4f}, '
            f'Loss: {loss:.4f} '
            f'({cor}/{tot})')
        for epoch in pbar:
            if args.use_pfi and epoch == args.pfi_epoch:
                # change to PFI
                # delayed to increase likelihood of convergence
                acc, loss, cor, tot = test(mdl, device, test_loader)
                tqdm.write(
                    f'Validation Accuracy ({epoch}): {acc:.4f}, '
                    f'Loss: {loss:.4f} '
                    f'({cor}/{tot})')

                pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
                inj_model = pfi_util.random_inj_per_layer()
                mdl = inj_model
                pbar.set_description('PFI Training')
            elif epoch % args.log_frequency == 0 and epoch != 0:
                # validation every N epochs
                # also check to see if PFI should be turned on now
                acc, loss, cor, tot = test(mdl, device, test_loader)
                tqdm.write(
                    f'Validation Accuracy ({epoch}): {acc:.4f}, '
                    f'Loss: {loss:.4f} '
                    f'({cor}/{tot})')

            pbar.set_postfix(lr=get_lr(optimizer), acc=f'{acc:.2f}%')

            if TERMINATE:
                if input('Really quit? [y]/n') != 'n':
                    logging.info('Terminating')
                    break
                else:
                    TERMINATE = False

            train(mdl, device, train_loader, optimizer)
            scheduler.step()

    pdb.set_trace()
    if not TERMINATE:
        # acc, conf = eval_confidences(mdl, device, test_loader)
        acc, _, _, _ = test(mdl, device, test_loader)
        mem, _, _, _ = test(mdl, device, train_loader)
    elif input('Evaluate? y/[n]') == 'y':  # only ask if terminated
        # acc, conf = eval_confidences(mdl, device, test_loader)
        acc, _, _, _ = test(mdl, device, test_loader)
        mem, _, _, _ = test(mdl, device, train_loader)
    else:
        conf = np.zeros(10)
        mem = -1.0

    print(f"""Final model accuracy: {acc:.2f}%
          Memorized: {mem:.3f}%
          Confidences: {conf}""")

    if args.save_model:
        if TERMINATE:
            # only ask if terminated
            if input('Early terminated, save model? y/[n]') != 'y':
                logging.warning("Didn't save")
                return
        torch.save(mdl.state_dict(), name)
        logging.info('Saved %s', name)


def signal_handler(sig, frame):
    '''Handle an interrupt for a graceful exit'''
    logging.warning('Interrupt caught: Exiting at end of epoch')
    global TERMINATE
    TERMINATE = True


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(level=logging.DEBUG)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--use-pfi', action='store_true', default=False,
                        help='Use PFI as a dropout alternative')
    parser.add_argument('--pfi-epoch', type=int, default=50,
                        help='Epoch at which to activate PFI')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-frequency', type=int, default=50)

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    main()
