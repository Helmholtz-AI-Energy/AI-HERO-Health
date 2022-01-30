from dataset import CovidImageDataset
from argparse import ArgumentParser
import torch
import torch.nn as nn
from model import VGG
import numpy as np
import os
from pytorch_lightning.utilities.seed import seed_everything
import random


def seed_worker(worker_id):
    '''
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    '''
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        train_loss += loss
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def val(model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += nn.CrossEntropyLoss()(output, target)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss / len(val_loader), correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/hkfs/work/workspace/scratch/im9193-health_challenge/data')
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--augment", type=str, default='resize_rotate_crop')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', help='saves the trained model')
    parser.add_argument('--model_name', type=str, help='model file name', default='vgg_baseline')

    args = parser.parse_args()

    device = torch.device("cuda")

    # the following 3 lines are only needed to make the training fully reproducible, you can remove them
    seed_everything(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

    data_base = args.data_dir

    trainset = CovidImageDataset(
        os.path.join(data_base, 'train.csv'),
        os.path.join(data_base, 'imgs'),
        transform=args.augment)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=128,
                                              worker_init_fn=seed_worker)
    valset = CovidImageDataset(
        os.path.join(data_base, 'valid.csv'),
        os.path.join(data_base, 'imgs'),
        transform=None)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=128,
                                            worker_init_fn=seed_worker)

    model = VGG('VGG19').cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print('CUDA available:', torch.cuda.is_available())

    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, trainloader, optimizer, epoch)
        val(model, device, valloader)
        scheduler.step()

    if args.save_model:
        model_dir = '/hkfs/work/workspace/scratch/im9193-health_challenge_baseline/saved_models'  # TODO adapt to your group workspace
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "{}.pt".format(args.model_name)))


if __name__ == '__main__':
    main()
