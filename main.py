import sys
import ast
import torch
from torch.optim import Adam
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import dgl
# --- 更换模型实现 change model ---
# from model import CG, CosineDecayScheduler, LogReg
from model_original_g import CG, CosineDecayScheduler, LogReg
# from model_original_g_recursion import CG, CosineDecayScheduler, LogReg

from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from dataset import load_graph_classification_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import hyperopt.pyll.stochastic

warnings.filterwarnings('ignore')


def load_best_configs(args, path):
    """
    """
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.dataname]
    for k, v in configs.items():
        # 已经有的参数不覆盖，注意默认参数
        if args.cmd_first and hasattr(args, k):
            continue
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(embeddings, labels):
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        f1 = f1_score(y_test, preds, average="micro")
        result.append(f1)
    test_f1 = np.mean(result)
    test_std = np.std(result)

    return test_f1, test_std


def collate_fn(batch):
    # graphs = [x[0].add_self_loop() for x in batch]
    graphs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    batch_g = dgl.batch(graphs)
    labels = torch.cat(labels, dim=0)
    return batch_g, labels


# 用time 获得随机数种子
# seed = int(time.time())
seed = int(33302)
seed_everything(seed)
# print(f"seed: {seed}")
parser = argparse.ArgumentParser(description="GraphOP")
# parser.add_argument("--dataname", type=str, default="ENZYMES")



# 命令行参数是否优先
args = parser.add_argument("--cmd_first", type=int, default=0, help="whether cmd args first")
parser.add_argument("--cuda", type=int, default=0)
parser.add_argument("--dataname", type=str, default="MUTAG")
args = parser.add_argument("--depth", type=int, default=1)
args = parser.add_argument("--rate", type=float, default=0.1, help="ring mask rate must be in [1/batch_size, 1]") # 当batch_size 256 的时候需要设置最小至少为0.004
args = parser.add_argument("--ring_width", type=int, default=1)
args = parser.add_argument("--contrast_with_central_nodes", type=int, default=0)
args = parser.add_argument("--epochs", type=int, default=100)
args = parser.add_argument("--loss_fn", type=str, default="sce")

args = parser.parse_args()

args = load_best_configs(args, "config.yaml")

dataname = args.dataname

# 环带宽度，取值范围为 1 ~ depth + 1
assert args.ring_width <= args.depth + 1 and args.ring_width >= 1, "ring_width must be in [1, depth + 1]"

graphs, (n_feat, num_classes) = load_graph_classification_dataset(dataname)
train_idx = torch.arange(len(graphs))
batch_size = 256
train_sampler = SubsetRandomSampler(train_idx)
train_loader = GraphDataLoader(graphs, collate_fn=collate_fn,
                               batch_size=batch_size,  shuffle=True)
eval_loader = GraphDataLoader(graphs, collate_fn=collate_fn, batch_size=batch_size,
                              shuffle=True)

device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

print(f"init-{args}")

def train(args):
    if type(args) == dict:
        args = argparse.Namespace(**args)
    if not(1 <= args.ring_width and args.ring_width <= args.depth + 1):
        return {"status": STATUS_FAIL}
    if type(args.ring_width) != int:
        args.ring_width = int(args.ring_width)
    if type(args.hidden) != int:
        args.hidden = int(args.hidden)
    if type(args.epochs) != int:
        args.epochs = int(args.epochs)
    print(f"valid-{args}")
    seed_everything(seed)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_file_name = ''
    for k, v in args.__dict__.items():
        log_file_name += f"{k}-{v}_"
    log_file_name = log_file_name[:-1]
    file = open(f"logs/{dataname}_{time_str}.csv", "w")
    file.write("epoch,loss\n")
    file_config = open(f"logs/{dataname}_{time_str}.txt", "w")
    file_config.write(f"{log_file_name}_seed-{seed}\n")
    file_config.close()

    # model = CG(n_feat, args.hidden, args.rate, 32, args.alpha, args.layer, args.depth, args.ring_width).to(device)
    model = CG(n_feat, args.hidden, args.rate, 32, args.alpha, args.layer, args.depth, args.ring_width, args.contrast_with_central_nodes, args.loss_fn).to(device)
    optimizer = Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.w)
    lr_scheduler = CosineDecayScheduler(args.lr, args.warmup, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # update momentum
        mm = 1 - mm_scheduler.get(epoch - 1)
        # mm = 0.99
        # update learning rate
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for g, _ in train_loader:
            g = g.to(device)
            loss = model(g, g.ndata["attr"])
            loss_tmp = loss.item()
            # print(f"Epoch: {epoch}, Loss: {loss_tmp}")
            file.write(f"{epoch},{loss_tmp}\n")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_network(mm)

    x_list = []
    y_list = []
    model.eval()
    for g, label in eval_loader:
        g = g.to(device)
        z1 = model.get_embed(g, g.ndata['attr'])
        y_list.append(label.numpy())
        x_list.append(z1.detach().cpu().numpy())
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    # print(args)
    print(f"Acc: {test_f1}, Std: {test_std}")
    file.write(f"{test_f1},{test_std}\n")
    file.close()
    # train_embs = torch.from_numpy(x)[idx_train].to(device)
    # test_embs = torch.from_numpy(x)[idx_test].to(device)
    #
    # train_lbls = torch.from_numpy(y)[idx_train].to(device)
    # test_lbls = torch.from_numpy(y)[idx_test].to(device)
    # b_xent = torch.nn.CrossEntropyLoss()
    # accs = []
    # for i in range(20):
    #
    #     log = LogReg(train_embs.shape[1], 2)
    #     opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=5e-4)
    #     if torch.cuda.is_available():
    #         log.to(device)
    #
    #     for _ in range(100):
    #         log.train()
    #         opt.zero_grad()
    #         logits = log(train_embs)
    #         loss = b_xent(logits, train_lbls)
    #         loss.backward()
    #         opt.step()
    #
    #     log.eval()
    #     logits = log(test_embs)
    #     preds = torch.argmax(logits, dim=1)
    #     f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')  # f1 score
    #     accs.append(f1.item() * 100)
    #
    # accs = np.array(accs)
    #
    # best_acc = accs.mean().item()
    # best_std = accs.std().item()
    # if accs.mean().item() > best_acc:
    #     best_acc = accs.mean().item()
    #     best_std = accs.std().item()
    #     best_epoch = epoch
    #
    # print('avg_f1: {0:.2f}, f1_std: {1:.2f}\n'.format(best_acc, best_std))
    return {'loss': -test_f1, 'status': STATUS_OK}


train(args)

sys.exit(0)

space = {
    # 'dataname': hp.choice("dataname", ["MUTAG", "PROTEINS", "REDDIT-BINARY", "NCI1", "REDDIT-MULTI5K", "COLLAB", "DD", "ENZYMES", "PTC_MR", "NCI109"] ),
    'dataname': args.dataname,
    'depth': hp.randint('depth', 16+1),
    # 'epochs': hp.choice('epochs', [100, 200, 300, 400]),
    # 'epochs': hp.quniform('epochs', 0, 800+1, 20),
    'epochs': hp.choice('epochs', [100]),
    'rate': hp.quniform('rate', 0.004, 0.104, 0.004),
    # 'cuda': hp.choice('cuda', [0, 1, 2, 3]),
    'cuda': args.cuda,
    'ring_width': hp.quniform('ring_width', 1, 16+1+1, 1),  # 要小于depth
    # 'out_hidden': hp.choice('out_hidden', [64]),
    # "out_hidden": 64,
    # "out_hidden": hp.quniform('out_hidden', 1, 512+1, 1),
    'hidden': hp.choice('hidden', [512]),
    # "hidden": hp.quniform('hidden', 1, 512+1, 1),
    'lr': hp.choice('lr', [1e-04, 1e-05, 1e-06]),
    # 'warmup': hp.choice('warmup', [100.0]),
    'warmup': hp.uniform('warmup', 0, 200.0),
    'w': hp.choice('w', [1e-04, 1e-05, 1e-06]),
    # 'alpha': hp.choice('alpha', [0.3]),
    'alpha': hp.uniform('alpha', 0, 1),
    'layer': hp.choice('layer', [1, 2, 3]),
    'contrast_with_central_nodes': hp.choice('contrast_with_central_nodes', [0, 1])
}

trails = Trials()
best = fmin(train, space, algo=tpe.suggest, max_evals=300, trials=trails)

# 更新config.yaml文件，config.yaml文件中的参数是最优参数,示例如下
"""
MUTAG:
  out_hidden: 64
  # hidden: 32
  hidden: 512
  epochs: 500
  lr: 1e-3
  warmup: 100
  w: 1e-5
  acc: 90.50+-8.4
  alpha: 0.3
  layer: 1
PROTEINS:
  out_hidden: 64
  # hidden: 32
  hidden: 512
  epochs: 100
  lr: 1e-5
  warmup: 100
  w: 1e-5
  acc: 75.83+-2.1
  alpha: 0.5
  layer: 2
"""
print(best)
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(type(best))
    config[args.dataname] = best
    config[args.dataname]['acc'] = -trails.best_trial['result']['loss']
    # 输出最佳结果
    # print(config[args.dataname])
    # 保存回config.yaml文件
    config = ast.literal_eval(str(config))
    print(yaml.dump(config))
    with open("config.yaml", "w") as f:
        f.write(yaml.dump(config))
        # print(yaml.dump(config))

    
    # config[args.dataname]['acc'] = trails.best_trial['result']['loss']


    # config[args.dataname] = {}





# print(best)

# PTC = np.array([[65.6, 67.1, 64.4, 63, 62.2, 61.6, 62.7, 61.9, 61.3]])
# NCI109 = np.array([[79.57, 79.7, 80.2, 79.1, 79.9, 79.7, 79.5, 79.4, 79.3]])
# ENZYMES = np.array([[58.5, 56.8, 55.9, 57.6, 58, 57, 58.3, 59.3, 58]])
# MUTAG = np.array([[86, 88.4, 90.5, 90.5, 88.9, 88.4, 87.8, 88.3, 88.9]])
# z = pd.DataFrame(np.concatenate([NCI109, MUTAG], axis=0),
#                  columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                  index=["NCI109", "MUTAG"])
# sns.set_palette("pastel")
# # sns.set(style='darkgrid')
# sns.lineplot(data=z.transpose(), markers=True, linewidth=6)
# plt.xlabel(r"$\beta$")
# plt.ylabel("Accuracy")
# # sns.despine()
# # plt.title("Graph Classification")
# plt.grid()
# plt.savefig("./A2.svg", dpi=800)
# plt.show()
