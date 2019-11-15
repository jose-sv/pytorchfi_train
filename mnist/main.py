'''Train, using pytorchfi instead as a fancy dropout'''
from __future__ import print_function
import argparse
import logging
import pdb
import os
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
    return {'acc': accuracy, 'tloss': test_loss, 'corr': correct, 'len':
            len(test_loader.dataset)}


def eval_confidences(model, device, test_loader):
    '''Calculate statistics on the confidences across a dataset'''
    # TODO calculate tolerance(top2-diff)
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


def try_resume(name, device):
    '''If an unfinished model exists, load and use it.

    Input: model

    Return: model

    Return the model to use, and the epoch to start training from'''
    model = ResNet18().to(device)
    estrt = 0

    # if unfinalized exists, attempt to load from that first
    if os.path.isfile('tmp.ckpt') and \
       input('Load from unfinalized? [y]/n') != 'n':
        prev = torch.load('tmp.ckpt')

    elif os.path.isfile(name):
        prev = torch.load(name)
    else:  # nothing to resume!
        logging.info('Nothing to resume')
        return model, estrt

    if prev['epoch'] != -1:  # didn't complete
        logging.info('Resuming from %s at epoch %i, acc %i', prev,
                     prev['epoch'], prev['acc'])
        model.load_state_dict(prev['net'])
        estrt = prev['epoch']
    else:
        logging.warning('Overwriting %s', name)

    return model, estrt


def main(args, name, use_cuda):
    '''Setup and iterate over training'''
    global TERMINATE

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

    model, estrt = try_resume(name, device)

    # TODO test try resume
    pdb.set_trace()

    # TODO @ma3mool please check whether optimizer must be updated when
    # switching to PFI (Issue #1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4,
                          momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1,
                            last_epoch=estrt)

    with trange(estrt, args.epochs + 1, unit='Epoch', desc='Training') as pbar:
        for epoch in pbar:
            if epoch % args.log_frequency == 0 or epoch == args.pfi_epoch:
                # test
                t_out = test(model, device, test_loader)

                # update
                # use a tmp model to prevent accidental overwrites
                torch.save({'net': model.state_dict(), 'acc': t_out['acc'],
                            'epoch': epoch}, 'tmp.ckpt')
                tqdm.write(f'Updated tmp.ckpt')

                tqdm.write(
                    f"Validation Accuracy ({epoch}): {t_out['acc']:.4f}, "
                    f"Loss: {t_out['tloss']:.4f} "
                    f"({t_out['corr']}/{t_out['len']})")

            if args.use_pfi and epoch == args.pfi_epoch:  # change model
                pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
                model = pfi_util.random_inj_per_layer()
                pbar.set_description('PFI Training')

            pbar.set_postfix(lr=get_lr(optimizer), acc=f"{t_out['acc']:.2f}%")

            if TERMINATE:
                if input('Quit? [y]/n') != 'n':
                    logging.info('Terminating')
                    break
                else:
                    TERMINATE = False

            train(model, device, train_loader, optimizer)
            scheduler.step()

    if not TERMINATE or input('Evaluate? y/[n]') == 'y':
        t_out, conf = eval_confidences(model, device, test_loader)
        # t_out = test(model, device, test_loader)
        m_out = test(model, device, train_loader)

        print(f"""Final model accuracy: {t_out['acc']:.2f}%
              Memorized: {m_out['acc']:.3f}%
              Confidences: {conf}""")

    if args.save_model:
        if TERMINATE and input('Early terminated, save model? y/[n]') != 'y':
            # only ask if terminated
            logging.warning("Didn't save")
            return
        torch.save({'net': model.state_dict(), 'acc': t_out['acc'],
                    'epoch': -1}, name)
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

    # TODO allow for model ckpt name/path specification

    # Training settings
    PARSER = argparse.ArgumentParser(description='PyTorch MNIST Example')
    PARSER.add_argument('--use-pfi', action='store_true', default=False,
                        help='Use PFI as a dropout alternative')
    PARSER.add_argument('--pfi-epoch', type=int, default=50,
                        help='Epoch at which to activate PFI')

    PARSER.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    PARSER.add_argument('--test-batch-size', type=int, default=128,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')

    PARSER.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 350)')
    PARSER.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    PARSER.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    PARSER.add_argument('--log-frequency', type=int, default=50)

    PARSER.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    ARGS = PARSER.parse_args()

    NAME = "cifar_resnet18.pt"
    if ARGS.use_pfi:
        NAME = f"pfi_{NAME}"
        logging.info('Using PFI from epoch %i', ARGS.pfi_epoch)

    USE_CUDA = not ARGS.no_cuda and torch.cuda.is_available()

    main(ARGS, NAME, USE_CUDA)
    # TODO migrate model off local server after training
