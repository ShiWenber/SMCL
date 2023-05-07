import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm
import copy
from functools import partial
import torch.nn.functional as F
import numpy as np

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

class GraphSAGE_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, embedding_size, 'mean')
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
        ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(),
            nn.PReLU(),
            nn.PReLU(),
        ])

        self.dp = nn.Dropout(p=0.2)

    def forward(self, g, x):
        # x = g.ndata['feat']
        if 'batch' in g.ndata.keys():
            batch = g.ndata['batch']
        else:
            batch = None

        h1 = self.convs[0](g, self.dp(x))
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](g, h1 + x_skip_1)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](g, h1 + h2 + x_skip_2)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()


class Encoder1(nn.Module):
    def __init__(self, in_dim, out_dim, p1, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.act = nn.ModuleList()
        for layer in range(num_layers):  # excluding the input layer
            self.act.append(nn.PReLU())
            if layer == 0 and num_layers == 1:
                self.conv.append(GraphConv(in_dim, out_dim))
                self.bn.append(BatchNorm(out_dim))
            elif layer == 0:
                self.conv.append(GraphConv(in_dim, hidden))
                self.bn.append(BatchNorm(hidden))
            else:
                self.conv.append(GraphConv(hidden, out_dim))
                self.bn.append(BatchNorm(out_dim))

        self.dp = nn.Dropout(p1)


    def reset_parameters(self):
        for i in range(self.num_layers):
            self.conv[i].reset_parameters()
            self.bn[i].reset_parameters()

    # ?
    def forward(self, graph, feat):
        h = self.dp(feat)
        for i, layer in enumerate(self.conv):
            h = layer(graph, h)
            h = self.bn[i](h)
            if self.num_layers > 1 and i == 0:
                h = self.act[i](h)
        return h

# def mask(g, x, num=10):
#     num_nodes = g.num_nodes()
#     perm = torch.randperm(num_nodes, device=x.device)
#     nodes = perm[: num]
#     # print(nodes)
#     neighbor_sets = [set(g.successors(n)) for n in nodes]
#     connected_nodes = set().union(*neighbor_sets)
#     connected_nodes = connected_nodes.union(set(nodes))
#     connected_nodes = np.array(list(connected_nodes), dtype=np.int64)
#     connected_nodes = connected_nodes.tolist()
#     connected_nodes = list(set(connected_nodes))
#     nodes=connected_nodes
#     return nodes

def mask(g, x, num=10):
    num_nodes = g.num_nodes()
    perm = torch.randperm(num_nodes, device=x.device)
    nodes = perm[: num]
    # print(nodes)
    connected_nodes = set()
    for n in nodes:
        succesors = g.successors(n)
        connected_nodes |= set(succesors.tolist())
    connected_nodes = torch.tensor(list(connected_nodes), dtype=torch.long, device=x.device)
    connected_nodes = torch.unique(connected_nodes)
    assert type(connected_nodes) == torch.Tensor
    mask_nodes = torch.cat([nodes, connected_nodes])
    mask_nodes = torch.unique(mask_nodes)

    return mask_nodes

# def mask(g, x, num=10):
#     # 生成随机节点列表
#     nodes = torch.randperm(g.num_nodes(), device=x.device)[:num]
    
#     # 获取所有节点的后继节点
#     successors = g.successors(torch.arange(g.num_nodes(), device=x.device))
    
#     # 选择采样节点的后继节点，并去除重复元素
#     connected_nodes = successors.index_select(0, nodes)
#     connected_nodes = torch.unique(connected_nodes)
#     assert type(connected_nodes) == torch.Tensor
#     mask_nodes = torch.cat([nodes, connected_nodes])
#     mask_nodes = torch.unique(mask_nodes)

#     return mask_nodes

# def mask(g, x, num, recur=2):
#     """_summary_

#     Args:
#         g (_type_): _description_
#         x (_type_): _description_
#         num (_type_): 节点数量
#         recur (int, optional): 邻居递归次数. Defaults to 2.

#     Raises:
#         NotImplementedError: _description_

#     Returns:
#         _type_: _description_
#     """
#     pass




class CG(nn.Module):
    def __init__(self, in_dim, out_dim, p1, rate, hidden, layers):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_dim, out_dim, p1, hidden, layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.reset_parameters()
        self.rate = rate
        # enc?
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn("sce", 1)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, graph, feat):
        mask_nodes = mask(graph, feat, 1)
        remainder_graph_feat = feat.clone()
        sub_graph=graph.subgraph(mask_nodes)
        sub_graph_feat=feat[mask_nodes]
        remainder_graph_feat[mask_nodes] = 0.0
        h1 = self.online_encoder(graph, remainder_graph_feat)
        with torch.no_grad():
            h2 = self.target_encoder(sub_graph, sub_graph_feat)
        # print(h1[mask_nodes].size())
        # print(h2.size())
        # print(h1[mask_nodes].mean(dim=0).size())
        # print(h2.detach().mean(dim=0).size())
        loss = self.criterion(h1[mask_nodes].mean(dim=0), h2.detach().mean(dim=0))
        # loss = self.criterion(h1[mask_nodes], h2.detach())

        return loss

    def get_embed(self, graph, feat):
        h1 = self.online_encoder(graph, feat)
        return h1.detach()