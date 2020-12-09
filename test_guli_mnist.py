'''
    test the angular softmax loss
    @author: Yuan Yang
    @date: 2017.05.28
'''
from __future__ import print_function
import os
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np

from guliloss import guli_fun
import sys
sys.path.insert(0, '..')
from module_set import mfm


def make_imbalance_weight(dataset):
    weight = [0]*len(dataset)
    for idx, item in enumerate(dataset):
        if item[1] == 3:
            weight[idx] = 0.05
        else:
            weight[idx] = 1.0
    return weight

def load_weight(model, state_dict):
    '''
        load weight from state_dict to model, skip those with invalid shape
    :param model:
    :param state_dict:
    :return:
    '''
    for name, param in model.named_parameters():
        # add fc2
        #if name == 'module.fc2.weight':
        #    print('*** manully load f2, fuck wuxiang')
        #    print('shape 1',param.size())
        #    print('shape 2',state_dict['module.fc2.1.weight'].size())
        #    param.data.copy_(state_dict['module.fc2.1.weight'])
        #    continue
        if not name in state_dict:
            print('--> weight {} does not exist in checkpoint'.format(name))
            continue

        if param.size() == state_dict[name].size():
            print('==> loading weight from ',name)
            param.data.copy_(state_dict[name])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--loadweight', default='', type=str, metavar='PATH',
                    help='load part compatiable weight')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# train_data
train_data =datasets.MNIST('data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

# imbalance weight for sampling
sample_weight = make_imbalance_weight(train_data)
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(train_data))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader( train_data,
    batch_size=args.batch_size, shuffle=False, sampler=sampler, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# settings here
margin = 4

# gradually reduce the beta
betas_1 = np.array([100, 50, 20, 10]) # 'soft' start
betas_2 = np.linspace(10.0, 5.0, args.epochs+1-4) # then large constraint
betas = np.hstack([betas_1, betas_2])

# set beta to fixed value
#betas = np.array([1.0]*(args.epochs+1))

in_feature = 2
out_feature = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)
        

        self.fc2_softmax = nn.Linear(2, 10, bias=False)
        
        self.guli_weight = torch.zeros(10, 2).cuda()
        torch.nn.init.normal(self.guli_weight, 0, 0.01)
        self.guli_weight = Variable(self.guli_weight)

    def forward(self, x, label):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        emb_f = self.fc2(x)

        out1 = guli_fun()(emb_f, self.guli_weight, label)
        out2 = self.fc2_softmax(emb_f)
        return F.log_softmax(out1), F.log_softmax(out2), emb_f

model = Net()

if args.loadweight:
    if os.path.isfile(args.loadweight):
        weight_dict = torch.load(args.loadweight)
        load_weight(model, weight_dict)
    else:
        print("=> no checkpoint found at '{}'".format(args.loadweight))

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        index_target = target.view(target.size(0), 1)
        
        # norm the weight
        w = model.guli_weight.data
        w_norm = torch.norm(w, 2, 1)
        w = w /( w_norm.unsqueeze(1) + 1e-6)
        model.guli_weight.data[:] = w

        out1, out2, emb_f = model(data, index_target)
        
        loss1 = 0.1*F.nll_loss(out1, target)
        loss2 = F.nll_loss(out2, target)
        loss = loss1 + loss2
        loss.backward()

        # better to clip the gradient, sometimes large margin softmax gets
        # large grad back
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss1.data[0], loss2.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        index_target = target.view(target.size(0), 1)

        # feed a empty Variable ..
        out1, out2, emb_f = model(data, index_target)
        test_loss += F.nll_loss(out2, target).data[0]
        pred = out2.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    # set beta
    #model.set_beta(betas[epoch-1])
    #print('set beta to ', betas[epoch-1])

    # adjust learning rate
    adjust_learning_rate(optimizer, epoch)

    train(epoch)
    test(epoch)
    torch.save(model.state_dict(), 'asoftmax_'+str(epoch)+'.pth')

    # visualize the result
    model.eval()
    embeds = []
    labels = []

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # feed a empty Variable ..
        index_target = target.view(target.size(0), 1)
        
        output = model(data, index_target)
        output = output[2].cpu().data.numpy()
        target = target.cpu().data.numpy()

        embeds.append(output)
        labels.append(target)

    embeds = np.vstack(embeds)
    labels = np.hstack(labels)
    
    # save data and label
    torch.save(embeds, 'feature_{}.pth'.format(epoch))
    torch.save(labels, 'label_{}.pth'.format(epoch))

    print('embeds shape ',embeds.shape)
    print('labels shape ',labels.shape)
    num = len(labels)
    names = dict()
    for i in range(10):
        names[i]=str(i)

    palette = np.array(sns.color_palette("hls", 10))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(embeds[:,0], embeds[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(embeds[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    fname = 'mnist-sphereface-%d-epoch.png'%(epoch)
    plt.savefig(fname)
