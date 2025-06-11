# config.py
import argparse
import numpy as np
import torch
import torchvision.transforms as trn
import torchvision.datasets as dset
from models.wrn import WideResNet
from utils.tinyimages_80mn_loader import TinyImages
import utils.svhn_loader as svhn
from utils.display_results import get_measures, print_measures

# ========== 1️⃣ 参数配置 ==========
parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--gamma', default=1, type=float)
parser.add_argument('--beta',  default=0.5, type=float)
parser.add_argument('--rho',   default=0.01, type=float)
parser.add_argument('--strength', default=0.01, type=float)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--iter', default=10, type=int)
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')

args = parser.parse_args()

# ========== 2️⃣ 随机种子 ==========
torch.manual_seed(1)
np.random.seed(args.seed)
torch.cuda.manual_seed(1)

# ========== 3️⃣ 数据集 ==========
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('../data/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('../data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('../data/cifarpy', train=False, transform=test_transform)
    num_classes = 100

ood_data = TinyImages(transform=trn.Compose([
    trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
    trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]))

train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

# ========== 4️⃣ 其他工具 ==========
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

# ========== 5️⃣ 模型 ==========
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
model_path = f'./models/{args.dataset}_wrn_pretrained_epoch_99.pt'
net.load_state_dict(torch.load(model_path))

# ========== 6️⃣ 工具函数 ==========
# 直接从 utils.display_results 导入
# get_measures, print_measures

# ========== 7️⃣ 其他 ==========
# 你也可以把 scheduler, optimizer 等放这里，看个人需要
