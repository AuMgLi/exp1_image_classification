import argparse
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from alexnet import AlexNet
from datasets import load_cifar10


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, eval_loader, opt):
    print('==> Start training...')

    summary_writer = SummaryWriter('./runs/' + str(int(time.time())))

    is_cuda = torch.cuda.is_available()
    model = AlexNet()
    if is_cuda:
        model = model.cuda()

    optimizer = optim.SGD(
        params=model.parameters(),
        lr=opt.base_lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    criterion = nn.CrossEntropyLoss()

    best_eval_acc = -0.1
    losses = AverageMeter()
    accuracies = AverageMeter()
    global_step = 0
    for epoch in range(1, opt.epochs + 1):
        # train
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            global_step += 1
            if is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), outputs.shape[0])
            summary_writer.add_scalar('train loss', loss, global_step)

            _, preds = torch.max(outputs, dim=1)
            acc = preds.eq(targets).sum().item() / len(targets)
            accuracies.update(acc)
            summary_writer.add_scalar('train acc', acc, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('==> Epoch: %d; Average Train Loss: %.4f; Average Train Acc: %.4f' %
              (epoch, losses.avg, accuracies.avg))

        # eval
        model.eval()
        losses.reset()
        accuracies.reset()
        for batch_idx, (inputs, targets) in enumerate(eval_loader):
            if is_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), outputs.shape[0])

            _, preds = torch.max(outputs, dim=1)
            acc = preds.eq(targets).sum().item() / len(targets)
            accuracies.update(acc)

        summary_writer.add_scalar('eval loss', losses.avg, global_step)
        summary_writer.add_scalar('eval acc', accuracies.avg, global_step)
        if accuracies.avg > best_eval_acc:
            best_eval_acc = accuracies.avg
            torch.save(model, './weights/best.pt')
        print('==> Epoch: %d; Average Eval Loss: %.4f; Average/Best Eval Acc: %.4f / %.4f' %
              (epoch, losses.avg, accuracies.avg, best_eval_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.01)
    opt = parser.parse_args()
    # print(opt)

    if not os.path.exists('./weights/'):
        os.mkdir('./weights/')

    train_loader, eval_loader = load_cifar10(batch_size=opt.batch_size)
    train(train_loader, eval_loader, opt)
