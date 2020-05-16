'''Train, using pytorchfi instead as a fancy dropout'''
from __future__ import print_function
import argparse
import logging
import os
import csv
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from pytorchfi import PyTorchFI_Util as pfi_util
from pytorchfi import PyTorchFI_Core as pfi_core
from resnet.models.resnet import ResNet18

# pylint: disable=C0103
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


def try_resume(name, device, use_cuda):
    '''If an unfinished model exists, load and use it.

    Input: model

    Return: model, epoch to start/resume from'''
    model = ResNet18().to(device)
    estrt = 0

    # if unfinalized exists, attempt to load from that first
    if os.path.isfile(name):
        prev = torch.load(name)
        try:
            if prev['pfi']:
                # change model so it can load PFI checkpoints correctly
                pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
                model = pfi_util.random_inj_per_layer()
                logging.info('Resuming PFI model')
        except KeyError:
            # before we started saving the PFI flag
            logging.warning('Old model found! Continuing, but could fail to '
                            'load')
        res = name

    else:  # nothing to resume!
        logging.info('%s not found', name)
        return model, estrt

    if prev['epoch'] != -1:  # didn't complete
        logging.info('Resuming from %s at epoch %i, acc %i', res,
                     prev['epoch'], prev['acc'])
    else:
        logging.info('Resuming from %s, acc %i', res, prev['acc'])

    model.load_state_dict(prev['net'])
    estrt = prev['epoch']

    return model, estrt


def main(args, load_name, save_name, use_cuda):
    '''Setup and iterate over training'''
    global TERMINATE  # noqa

    progress = []

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

    model, estrt = try_resume(load_name, device, use_cuda)

    if args.use_pfi and args.pfi_epoch == 0 and estrt == 0:
        pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
        model = pfi_util.random_inj_per_layer()
        logging.info('Using PFI from epoch 0')

    # TODO @ma3mool please check whether optimizer must be updated when
    # switching to PFI (Issue #1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4,
                          momentum=0.9)

    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    desc = 'PFI Training' if args.use_pfi else 'Training'

    for _ in range(estrt):
        scheduler.step()

    with trange(estrt, args.epochs + 1, unit='Epoch', desc=desc) as pbar:
        for epoch in pbar:
            if epoch % args.log_frequency == 0 or epoch == estrt:
                t_out = test(model, device, test_loader)
                if not args.no_mem:
                    m_out = test(model, device, train_loader)
                    progress.append({'epoch': epoch, 'val_acc': t_out['acc'],
                                     'mem': m_out['acc'],
                                     'lr': get_lr(optimizer)})
                else:
                    m_out['acc'] = 'NA'
                    progress.append({'epoch': epoch, 'val_acc': t_out['acc'],
                                     'mem': -1, 'lr': get_lr(optimizer)})

                # update statistics and checkpoint
                tqdm.write(
                    f"Validation Accuracy ({epoch}): {t_out['acc']:.4f}, "
                    f"({t_out['corr']}/{t_out['len']}) "
                    f"Memorized: {m_out['acc']}, ")

            # change model to PFI at [epoch]
            if args.use_pfi and epoch == args.pfi_epoch and epoch != 0:
                pfi_core.init(model, 32, 32, 128, use_cuda=use_cuda)
                model = pfi_util.random_inj_per_layer()

                # make a new optimizer on the pfi model!
                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                      weight_decay=5e-4, momentum=0.9)
                scheduler = MultiStepLR(optimizer, milestones=[150, 250],
                                        gamma=0.1)
                for _ in range(epoch):  # update LR progress
                    scheduler.step()

                tqdm.write(f'Changed to PFI at {epoch}')
                pbar.set_description('PFI Training')

            pbar.set_postfix(lr=get_lr(optimizer), acc=f"{t_out['acc']:.2f}%")

            if TERMINATE:
                if input('Quit? [y]/n ') != 'n':
                    logging.info('Terminating')
                    break
                else:
                    TERMINATE = False

            train(model, device, train_loader, optimizer)
            scheduler.step()

    if not TERMINATE or input('Evaluate? y/[n] ') == 'y':
        t_out = test(model, device, test_loader)
        m_out = test(model, device, train_loader)
        progress.append({'epoch': epoch, 'val_acc': t_out['acc'],
                         'mem': m_out['acc'], 'lr': get_lr(optimizer)})

    print(f"Final model accuracy: {progress[-1]['val_acc']:.2f}%\n"
          f"Memorized: {progress[-1]['mem']:.3f}%")

    if args.no_save:
        if TERMINATE and input('Early terminated, save model? y/[n] ') != 'y':
            # only ask if terminated
            logging.warning("Didn't save")
            return None
        torch.save({'net': model.state_dict(), 'acc': t_out['acc'],
                    'epoch': -1 if not TERMINATE else epoch,
                    'pfi': epoch > args.pfi_epoch and args.use_pfi}, save_name)
        logging.info('Saved %s', save_name)

    return progress


def signal_handler(sig, frame):
    '''Handle an interrupt for a graceful exit'''
    logging.warning('Caught Signal %s: Exiting at end of epoch', sig)
    logging.debug('Caught at %s', frame)
    global TERMINATE  # noqa
    TERMINATE = True


if __name__ == '__main__':
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    logging.basicConfig(level=logging.DEBUG)

    # TODO allow for model ckpt name/path specification

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    pfi_group = parser.add_argument_group('PFI Options')
    pfi_group.add_argument('--use-pfi', action='store_true', default=False,
                           help='Use PFI as a dropout alternative')
    pfi_group.add_argument('--no-mem', action='store_true', default=False,
                           help='Traing quickly, without testing memorization')
    pfi_group.add_argument('--pfi-epoch', default=0, type=int,
                           help='Epoch from which to start PFI')

    ckpt_group = parser.add_argument_group('Checkpoint Options')
    ckpt_group.add_argument('--load-name', default='cifar_resnet18', type=str,
                            help='Checkpoint to use as base for training')
    ckpt_group.add_argument('--save-name', default='cifar_resnet18', type=str,
                            help='Checkpoint to save trained model to')
    ckpt_group.add_argument('--no-save', action='store_false', default=True,
                            help='For Saving the current Model')

    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--batch-size', type=int, default=128,
                             metavar='N',
                             help='batch size for training (default: 64)')
    train_group.add_argument('--test-batch-size', type=int, default=128,
                             metavar='N',
                             help='batch size for testing (default: 1000)')

    train_group.add_argument('--epochs', type=int, default=350, metavar='N',
                             help='number of epochs to train (default: 350)')
    train_group.add_argument('--lr', type=float, default=0.1, metavar='LR',
                             help='learning rate (default: 1.0)')
    train_group.add_argument('--no-cuda', action='store_true', default=False,
                             help='disables CUDA training')
    train_group.add_argument('--log-frequency', type=int, default=50)

    args = parser.parse_args()

    SAVE_NAME = f"{args.save_name}.pt"
    if args.use_pfi:
        SAVE_NAME = f"pfi_{SAVE_NAME}"
        logging.info('Using PFI from epoch %i', args.pfi_epoch)

    USE_CUDA = not args.no_cuda and torch.cuda.is_available()

    progress = main(args, f'{args.load_name}.pt', SAVE_NAME, USE_CUDA)
    # TODO migrate model off local server after training

    if progress is not None:
        with open(f'{SAVE_NAME}_train.log', 'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=['epoch', 'val_acc',
                                                          'mem', 'lr'])
            writer.writeheader()
            for log in progress:
                writer.writerow(log)
