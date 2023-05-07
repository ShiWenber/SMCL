import torch
import dgl
from torch.optim import AdamW, Adam
from model import CG, CosineDecayScheduler
from torch_geometric import seed_everything
import numpy as np
import warnings
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from eval import label_classification, fit_ppi_linear
from dataset import load_data, get_ppi
import time

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    configs = configs[args.dataname]

    for k, v in configs.items():
        if "lr" in k or "w" in k:
            v = float(v)
        setattr(args, k, v)
    return args


seed_everything(35536)
parser = argparse.ArgumentParser(description="SimGOP")
parser.add_argument("--dataname", type=str, default="Cora")
parser.add_argument("--cuda", type=int, default=0)
args = parser.parse_args()
args = load_best_configs(args, "config.yaml")
dataname = args.dataname
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
label_type = args.label
graph, feat, label, train_mask, val_mask, test_mask = load_data(dataname)
n_feat = feat.shape[1]
n_classes = np.unique(label).shape[0]
graph = graph.to(device)
label = label.to(device)
feat = feat.to(device)

time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
# log_file = open(f"logs/{dataname}_mask1.csv", "w")

log_file = open(f"logs/{dataname}_mask_{time_str}.csv", "w")

log_file.write("epoch,loss\n")


def train():
    # 记录开始时间
    start_time = time.time()
    # 学习率动态更新？
    model = CG(n_feat, args.dim, args.p1, args.rate, args.hidden, args.layer).to(device)
    lr_scheduler = CosineDecayScheduler(args.lr, args.warmup, args.epochs)
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, args.epochs)
    optimizer = Adam(model.trainable_parameters(), lr=args.lr, weight_decay=args.w)

    for epoch in range(1,args.epochs+1):
        model.train()
        lr = lr_scheduler.get(epoch - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        mm = 1 - mm_scheduler.get(epoch - 1)

        loss = model(graph, feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        loss_tmp = loss.item()
        print(f"Epoch: {epoch}, Loss: {loss_tmp}")
        log_file.write(f"{epoch},{loss_tmp}\n")

    # 这里的检验
    z1 = model.get_embed(graph, feat)
    acc = label_classification(z1, train_mask, val_mask, test_mask,
                               label, label_type, name=dataname, device=device)
    # print(acc)
    print(f" Acc: {acc['Acc']['mean']}, Std: {round(acc['Acc']['std'], 4)}")
    end_time = time.time()
    log_file.write(f"{acc['Acc']['mean']},{round(acc['Acc']['std'], 4)}\n")
    log_file.write(f"{end_time - start_time},0\n")
    log_file.close()

train()