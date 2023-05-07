# import torch
# import dgl
# from torch.optim import AdamW, Adam
# from model import CG, CosineDecayScheduler
# from torch_geometric import seed_everything
# import numpy as np
# import warnings
# import yaml
# import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from eval import label_classification, fit_ppi_linear
# import dgl
# from dataset import load_data, get_ppi
# graph, feat, label, train_mask, val_mask, test_mask = load_data("Cora")
#
# def mask(g, x, num=1):
#     num_nodes = g.num_nodes()
#     perm = torch.randperm(num_nodes, device=x.device)
#     nodes = perm[: num]
#     print(nodes)
#     neighbor_sets = [set(g.successors(n)) for n in nodes]
#     connected_nodes = set().union(*neighbor_sets)
#     connected_nodes = connected_nodes.union(set(nodes))
#     connected_nodes = np.array(list(connected_nodes), dtype=np.int64)
#     connected_nodes = connected_nodes.tolist()
#     connected_nodes = list(set(connected_nodes))
#     nodes=connected_nodes
#     print(nodes)
#     return nodes
#
# def get():
#     # 创建一个有向图
#     g = dgl.DGLGraph()
#     g.add_nodes(5)
#     g.add_edges([0, 1, 1, 2, 3], [1, 2, 3, 0, 4])
#     nodes = [1, 2]
#     # successors(v, etype='X')表示获取出发节点为v, 边类型为‘X’的终止节点
#     neighbor_sets = [set(g.successors(n)) for n in nodes]
#     connected_nodes = set().union(*neighbor_sets)
#     connected_nodes = connected_nodes.union(set(nodes))
#     connected_nodes = np.array(list(connected_nodes), dtype=np.int64)
#     connected_nodes = connected_nodes.flatten()
#     connected_nodes = connected_nodes.tolist()
#     connected_nodes=list(set(connected_nodes))
#     print(connected_nodes)
# g = dgl.DGLGraph()
# g.add_nodes(5)
# g.add_edges([0, 1, 1, 2, 3], [1, 2, 3, 0, 4])
# sub=g.subgraph(mask(g,g,1))
#
# print(sub)
#
# nodes=mask(graph,feat)
# print()
# print(len(nodes))
import torch

# 创建一个形状为(6, 128)的tensor
x = torch.randn(3, 5)
print(x)
# 计算第0维上的均值
mean = x.mean(dim=0)
print(mean)
# 将均值扩展为形状为(1, 128)的tensor
mean = mean.unsqueeze(0)
print(mean)
# 输出结果
print(mean.size())
print(mean)
