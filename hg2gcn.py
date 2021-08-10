## Zijian Liu
## "Official PyTorch Source of the paper - Residual hypergraph neural network"
## time - Aug 10th, 2021
## email: liuzijianupc@sina.cn
# Details of the network and preprocessing will be given after the article is received

import argparse # 导入模块
parser = argparse.ArgumentParser()
# parser.add_argument('-w', '--weights', default="YOLO_small.ckpt", type=str)
# parser.add_argument('--data_dir', default="data", type=str)
# parser.add_argument('--threshold', default=0.2, type=float)
parser.add_argument('--seed', default=2021, type=int)
parser.add_argument('--data', default='cocitation', type=str)
parser.add_argument('--dataset', default='cora', type=str)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--rate', default=1e-3, type=bool)
parser.add_argument('--decay', default=1e-5, type=bool)
parser.add_argument('--epoch', default=400, type=bool)
parser.add_argument('--gpu_id', default="1", type=str)
parser.add_argument('--depth', default=2, type=int)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--no_train', default=True, action='store_false')
args = parser.parse_args()

# seed
import os, torch, numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import scipy.sparse as sp
# from sklearn import manifold
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import random
# matplotlib inline
# config InlineBackend.figure_format='retina'
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# from tqdm import tqdm
import warnings
 
warnings.filterwarnings("ignore")

setup_seed(args.seed)

# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ['PYTHONHASHSEED'] = str(args.seed)

class gcn_layer(nn.Module):
    def __init__(self, in_c, out_c, alpha=args.alpha):
        super(gcn_layer, self).__init__()
        # is coming soon...
        pass

    def forward(self, x, adj):
        # is coming soon...
        pass
        return x


class gcn(nn.Module): # citeseer
    def __init__(self, in_c, out_c, hid_c=128):
        super(gcn, self).__init__()
        # is coming soon...
        pass

    def forward(self, x, adj):
        # is coming soon...
        pass
        return x


def calc_adj(dataset):
    # is coming soon...
    pass
    return H_data, indices


def train_step(model, x, pair_H, train, test, y, opt, args):
    best_acc = 0.0
    J = []
    L = []
    for epoch in range(args.epoch):
        model.train()
        # scheduler.step()
        output = model(x, pair_H)[-1]
        opt.zero_grad()
        loss = F.cross_entropy(output[train], y[train])
        J.append(loss.item())
        loss.backward()
        opt.step()
        if epoch % 10 == 9:
            model.eval()
            output = model(x, pair_H)[-1]
            acc = torch.eq(output[test].argmax(dim=1),y[test]).sum().item() / test.__len__()
            print("Epoch: {:03d}, loss: {:.6f}, acc: {:.4f}".format(
                epoch+1, loss.item(), acc )
                )
            # best_acc = acc
            L.append(acc) 
            if best_acc < acc:
                best_acc = acc
                torch.save(model, 'hgcn_{}_{}.pth'.format(args.data, args.dataset))

    print('Best acc: {:.4f}'.format(best_acc) )


def test_step(model, x, pair_H, test, y, args):
    
    with torch.no_grad():
        model.eval()
        output = model(x, pair_H)
        acc = torch.eq(output[-1][test].argmax(dim=1),y[test]).sum().item() / test.__len__()
        print('Best acc: {:.4f} on {} - {}'.format(acc, args.data, args.dataset))


def main():
    is_train = args.no_train
    # is_test = False
    print(args)

    # load data
    from data import data
    dataset, train, test = data.load(args) # cora 140, pubmed 78, citeseer 138, dblp (1740, 39562)
    print("length of train is", len(train))
    x, y = dataset['features'], dataset['labels']

    H_data, indices = calc_adj(dataset)

    model = gcn(x.shape[1], y.shape[1])
    if not is_train:
        model = torch.load('hgcn_{}_{}.pth'.format(args.data, args.dataset))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    x = torch.FloatTensor(np.array(x)).cuda()
    y = torch.LongTensor(y.argmax(axis=1)).cuda()

    pair_H = torch.sparse_coo_tensor(indices=indices, values=H_data, size=(x.shape[0],x.shape[0])).cuda()
    model = model.cuda()
    if is_train:
        train_step(model, x, pair_H, train, test, y, opt, args)
    else:
        test_step(model, x, pair_H, test, y, args)

if __name__ == "__main__":
    main()

# Details of the network and preprocessing will be given after the article is received