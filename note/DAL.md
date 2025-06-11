# DAL

1. Code structure

   ```
   .
   ├── README.md
   ├── data
   ├── main.py
   ├── models
   │   ├── cifar100_wrn_pretrained_epoch_99.pt
   │   ├── cifar10_wrn_pretrained_epoch_99.pt
   │   └── wrn.py
   └── utils
       ├── 80mn_cifar_idxs.txt
       ├── display_results.py
       ├── svhn_loader.py
       └── tinyimages_80mn_loader.py
   
   4 directories, 9 files
   (base)
   ```

   

2. hyper params

   ```python
   # dataset															选择是 cifar10 还是 cifar100 作为主 in-distribution 数据集。
   # epochs															总共训练的轮数。越大，训练越充分，但也可能过拟合。
   # learning_rate (-lr)									初始学习率。决定每一步梯度下降的幅度。
   # batch_size													in-distribution (CIFAR) 的 batch 大小。
   # oe_batch_size												OOD (TinyImages) 的 batch 大小。
   # test_bs															测试阶段的 batch 大小。
   # momentum														SGD 动量参数，帮助平滑梯度更新。
   # decay (-d)													权重衰减（L2正则化），防止过拟合。
   ---------------------------------------------------------------------------------------------------
   parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                       help='Choose between CIFAR-10, CIFAR-100.')
   
   # Optimization options
   parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
   parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
   parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
   parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
   parser.add_argument('--test_bs', type=int, default=200)
   parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
   parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
   # WRN Architecture
   parser.add_argument('--layers', default=40, type=int, help='total number of layers')
   parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
   parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
   # DAL hyper parameters
   parser.add_argument('--gamma', default=1, type=float)
   parser.add_argument('--beta',  default=0.5, type=float)
   parser.add_argument('--rho',   default=0.01, type=float)
   parser.add_argument('--strength', default=0.01, type=float)
   parser.add_argument('--warmup', type=int, default=0)
   parser.add_argument('--iter', default=10, type=int)
   # Others
   parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
   parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
   ```

   

3. init env

   ```bash
   # download conda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   
   # install
   bash miniconda.sh
   
   # activate conda
   source ~/miniconda3/etc/profile.d/conda.sh
   
   # update conda
   conda update -n base -c defaults conda -y
   
   # create env
   conda create -n myenv python=3.7 / conda env create -f env.yml (optional)
   
   # create jupyter kernel (-y means reply yes automatically)
   conda install ipykernel -y 
   
   # register kernel
   python -m ipykernel install --user --name dal-env --display-name "DAL Env"
   
   # download tree
   sudo apt install tree -y
   ```



4. use tinydataset

   ```python
   from utils.tinyimages_80mn_loader import TinyImages
   
   
   dataset = TinyImages(transform=transform, exclude_cifar=False)
   dataset.__len__ = lambda: 1000000  # 改写长度为 1M
   ```

   

5. history

   ```shell
   # modified tinyimage dataloader
   # modified main.py
   # add log and data model checkpoint
   python main.py cifar10 --gamma=10 --beta=.01  --rho=10  --iter=10 --learning_rate=0.07 --strength=1
   
   # issue, need to set download = True, or 
   train_data_in = dset.CIFAR10('../data/cifarpy', train=True, transform=train_transform, download=True) 
   
   ```


6. extra dataset

   ```shell
   # dtd
   wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
   
   # iSUN (provided by prof)
   wget "https://dl.dropboxusercontent.com/scl/fi/wpkzixs1zbqomg5ufq0dd/iSUN.tar.gz?rlkey=46mty3ly8kk3vdxtlnmdjc6zu" -O iSUN.tar.gz
   
   # try another iSUN
   
   
   
   ```

   

