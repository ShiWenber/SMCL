import random
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.glob import SumPooling, MaxPooling, AvgPooling
from dgl.nn.pytorch import GINConv
from torch_geometric.nn import BatchNorm, LayerNorm
import copy
import numpy as np
import functools


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


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


# @functools.lru_cache(maxsize=16)
def get_all_neighbors(graph:dgl.DGLGraph, nodes:torch.Tensor, depth:int, adj_matrix:torch.Tensor=None):
    """获得邻居节点集合张量（逻辑上是集合，物理数据结构为torch.Tensor），且去重
    Args:
        graph (dgl.DGLGraph): 输入图
        nodes (torch.Tensor): 中心节点集合张量，1*nodes_num
        depth (int): 递归深度，为0时相当于点遮盖，直接返回nodes
    Returns:
        torch.Tensor: 所有中心点的邻居节点组成的无重复节点张量, 1*neighbors_num
        torch.Tensor: 每层外扩获得的邻居节点组成的无重复节点张量, depth*neighbors_num，0位置是第一层的邻居，1位置是第二层的邻居，以此类推
    """
    if adj_matrix == None:
        adj_matrix = graph.adjacency_matrix().to(nodes.device)
    if depth < 0:
        return torch.Tensor([]).to(nodes.device), []
    if depth == 0:
        return nodes, []
    else:
        level_neighbors = [] # 用于存储每一层的邻居
        level_neighbors.append(nodes) # 用level_neighbors[0]存储中心点
        one_hot_central_nodes = torch.eye(adj_matrix.shape[0])[nodes]

        # 因cuda内存不足，所以采用稀疏矩阵
        sparse_one_hot_matrix = one_hot_central_nodes.to_sparse()
        one_hot_central_nodes = sparse_one_hot_matrix.to(nodes.device)
        # current_mat = one_hot_central_nodes @ adj_matrix
        # current_mat = torch.sparse.mm(adj_matrix, one_hot_central_nodes.T)
        
        current_mat = torch.sparse.mm(adj_matrix, one_hot_central_nodes.t())
        # 如果current_mat是两个稀疏矩阵相乘，那么current_mat.nonzero()报错
        current_mat = current_mat.to_dense()
        current_neighbors = current_mat.nonzero()
        # print('current_neighbors: ', current_neighbors)
        neighbors = current_neighbors[:,0]
        neighbors = torch.cat((neighbors, level_neighbors[0]), dim=0)
        neighbors = torch.unique(neighbors)
        # neighbors = torch.unique(neighbors)
        level_neighbors.append(neighbors)
        for i in range(2, depth + 1):
            current_mat = torch.sparse.mm(adj_matrix, current_mat)
            current_neighbors = current_mat.nonzero()
            neighbors = current_neighbors[:,0]
            neighbors = torch.cat((neighbors, level_neighbors[i - 1]), dim=0)
            neighbors = torch.unique(neighbors)
            # neighbors = torch.masked_select(neighbors, torch.logical_not(torch.isin(neighbors, level_neighbors[i - 1])))
            level_neighbors.append(neighbors)
        assert neighbors.is_cuda == True
        # print("level_neighbors: ", level_neighbors)
        # print("central_nodes: ", nodes)
        return neighbors, level_neighbors

def mask(g:dgl.DGLGraph, x:torch.Tensor, num, depth, ring_width, central_nodes=None):
    """遮盖子图，返回遮盖节点集合和中心节点集合
    1. 此处采用的方式是将两个圆形子图做集合差，得到环形子图
    2. 还存在一种方式是将多个不同距离的邻居集合做集合并，得到环形子图
    采用 1 会出现如下问题：
        对于图:
        1-2-3
        1-3
        从 1 出发到 3 有两条路径，当我们取 depth = 2，ring_width = 1 时，我们得到 Mask_nodes 为空集
        这可能会导致遮盖的点集较少，且无法成为一个严格的环
    采用 2 会出现如下问题：
        对于图:
        1-2-3-4-5
        从 1 出发，当我们取 depth = 4，ring_width = 1 时，计算距离为2的邻居时可能会走“回头路”（1->2->1）这会导致出现一些很奇怪的现象，找距离固定的邻居时候出现回头，同一距离的邻居点形成类似水波涟漪的同心环结构，遮盖的点稀疏且有一定规律，利于重建
        对于图:
        1-2 --- 3 --- 4
         \  \   /   /
          \ 7 - 5 /
            ---8-- 
        从 1 出发，当我们取 depth = 3，ring_width = 1 时，计算距离为2的邻居时可能会走“回头路”，导致 2345 全部被遮，这会导致被遮盖的点过于集中，遮盖的时候丢失过多信息
    综上，我们认为 1 更加稳定一些，对 2 来说，同一种方法在不同的图上表现出的效果各不相同，可以作为未来的尝试
    Args:
        g (dgl.DGLGraph): 输入图
        x (torch.Tensor): 特征张量
        num (int): 随机遮盖的子图数量（中心点数量）
        depth (int, optional): 遮盖节点的邻居深度. Defaults to 1.
        num_ring (int): batch中遮盖的环形的数量. Defaults to 1. 范围为[0, batch_size]
    Returns:
        torch.Tensor: mask_nodes, 遮盖节点集合
        torch.Tensor: central_nodes, 中心节点集合，可以去除，仅用在需要查看中心节点特征的情况下，猜测当gcn层数为3时，递归深度低于3就可能导致中心节点特征丢失 todo
    """
    # print('masked_ring_num:',num)
    assert 1 <= ring_width <= depth + 1, "ring_width must be in [1, depth + 1]"
    assert num >= 0 and num <= g.batch_size, "num_ring must be in [0, batch_size]"
    num_nodes = g.num_nodes()
    adj_matrix = g.adjacency_matrix().to(x.device)


    central_nodes = None
    if num == 0:
        return torch.Tensor([]).to(x.device), torch.Tensor([]).to(x.device)
    elif num == 1:
        perm = torch.randperm(num_nodes, device=x.device)
        central_nodes = perm[: 1]   
    else:
        batch_num_nodes = g.batch_num_nodes() # batch_size * 1 的张量，每个元素是batch中一个小图的节点数
    
        # 生成一个idx，用于索引batched graph中的节点
        idx = torch.zeros(len(batch_num_nodes) + 1, dtype=torch.int64)
        for i in range(1, len(batch_num_nodes) + 1):
            idx[i] = idx[i - 1] + batch_num_nodes[i - 1]
        random_idx = torch.zeros(len(batch_num_nodes), dtype=torch.int64)
        for i in range(0, len(idx) - 1):
            # print(idx[i], idx[i + 1])
            # 生成一个 batch_num_nodes[i] - batch_num_nodes[i + 1] 之间的随机数
            random_idx[i] = random.randint(idx[i], idx[i + 1] - 1)

        # 从random_idx中随机选取num个中心点
        central_nodes = random_idx[torch.randperm(len(random_idx), device=x.device)[: num]]
        # print('central_nodes:',central_nodes.shape)
        # print('central_nodes:',central_nodes)
    

    # print(central_nodes.device)
    assert type(central_nodes) == torch.Tensor
    # 判断central_nodes的shape为1*num
    central_nodes = central_nodes.to(x.device)
    # print(nodes)
    assert central_nodes.is_cuda == True

    # 外圆节点集合(含中心点)
    connected_nodes_out, level_neighbors = get_all_neighbors(g, central_nodes, depth, adj_matrix)
    assert type(connected_nodes_out) == torch.Tensor
    mask_nodes_out = torch.cat((central_nodes, connected_nodes_out), dim=0)
    mask_nodes_out = torch.unique(mask_nodes_out)

    # 内圆节点集合(含中心点)
    if (depth - ring_width) >= 0:
        not_mask_nodes = level_neighbors[depth - ring_width]
    else:
        not_mask_nodes = torch.Tensor([]).to(x.device) 



    # not_mask_nodes = get_all_neighbors(g, central_nodes,depth - ring_width, adj_matrix)
    # not_mask_nodes = torch.cat((central_nodes, not_mask_nodes), dim=0)
    # not_mask_nodes = torch.unique(not_mask_nodes)

    # 环形应遮盖节点集合，有于torch的集合操作，返回的mask_nodes一定是无重复的
    # mask_nodes = torch.masked_select(mask_nodes_out, torch.isin(mask_nodes_out, not_mask_nodes))
    mask_nodes = torch.masked_select(mask_nodes_out, torch.logical_not(torch.isin(mask_nodes_out, not_mask_nodes)))

    return mask_nodes, central_nodes



class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Encoder1(nn.Module):
    def __init__(self, in_hidden, out_hidden, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers):  # excluding the input layer
            if layer == 0:
                mlp = MLP(in_hidden, out_hidden, hidden)
            else:
                mlp = MLP(hidden, out_hidden, hidden)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(BatchNorm(hidden))
            
            self.act.append(nn.PReLU())
        self.pool = SumPooling()

    def forward(self, graph, h):
        # print('h:',h)
        output = []
        for i, layer in enumerate(self.ginlayers):
            # # 如果h只有一个样本，那么就进行unsqueeze(0)操作，增加一个维度
            # # print('h.dim():',h.dim())
            # # print('h',h)
            

            assert h.shape[0] != 1, 'h只有单样本，无法使用batch_norm归一化'
            h = layer(graph, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            output.append(self.pool(graph, h))

        return h, torch.cat(output, dim=1)

    # def reset_parameters(self):
    #     self.conv1.reset_parameters()
    #     self.conv2.reset_parameters()
    #     self.conv3.reset_parameters()
    #     self.bn.reset_parameters()
    #     self.bn2.reset_parameters()
    #     self.bn3.reset_parameters()


class CG(nn.Module):
    def __init__(self, in_hidden, out_hidden, rate, hidden, alpha, layers, depth, ring_width, mask_central_nodes):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_hidden, out_hidden, hidden, layers)
        self.target_encoder = Encoder1(in_hidden, out_hidden, hidden, layers)
        # self.decoder = Encoder1(hidden, out_hidden, in_hidden, 1)
        self.rate = rate
        self.alpha = alpha
        self.depth = depth
        self.ring_width = ring_width
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        self.criterion = self.setup_loss_fn("sce")
        self.contrast_with_central_nodes = mask_central_nodes

        
        self.hidden = hidden

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # self.proj_head = nn.Sequential(nn.Linear(out_hidden * layers, 64 * layers), nn.ReLU(inplace=True),
        #                                nn.Linear(64 * layers, 128))

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def cl_loss(self, x, x_aug, t=0.2):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss

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
        # 将num_ring转为int
        num_ring = int(self.rate * graph.batch_size)
        # 如果num_ring为0，那么num_ring=1
        if num_ring == 0:
            num_ring = 1
        ring_nodes, central_nodes = mask(graph, feat, num_ring, self.depth, self.ring_width)
        # # print('mask_nodes:',mask_nodes)
        masked_graph_feat = feat.clone()
        # print('mask_nodes:',mask_nodes)
        # sub_graph=graph.subgraph(mask_nodes)
        # print('sub_graph:',sub_graph)
        # # print('sub_graph:',sub_graph)
        # sub_graph_feat=feat[mask_nodes]
        # # print('sub_graph:',sub_graph)
        
        # todo 这里遮蔽全图或者只遮蔽环可能效果不一致，可以尝试
        # mask_nodes = torch.cat((ring_nodes, central_nodes), dim=0)
        mask_nodes = ring_nodes

        if mask_nodes.shape[0] == 0:
            # todo 这里没有异常处理，如果mask_nodes为空，会导致后面的代码报错
            pass
            # return 
        else:
            masked_graph_feat[mask_nodes] = 0.0
            masked_graph_feat[mask_nodes] += self.enc_mask_token

        h1,_ = self.online_encoder(graph, masked_graph_feat)
        with torch.no_grad():
            # if sub_graph_feat.shape[0] == 1:
            #     # TODO 添加编码过程，将特征维度补全至和online_encoder一致
            #     # print('h只有单样本，无法使用batch_norm归一化') # 卷积直接报错？应该按照gin的特征来手动补全h2
            #     # print('sub_graph_feat:',sub_graph_feat)
            #     # 将特征复制到和hidden维度一致
            #     # TODO
            #     # h2 = sub_graph_feat.repeat(1, self.hidden // sub_graph_feat.shape[1]) # 1 * 28
            # else:
            #     h2,_ = self.target_encoder(sub_graph, sub_graph_feat)
            h2,_ = self.target_encoder(graph, feat)

        # 对比节点集合
        contrast_nodes = None
        if (self.contrast_with_central_nodes):
            # 对比学习的时候对比子图
            contrast_nodes = torch.cat((ring_nodes, central_nodes), dim=0)
        else:
            # 对比学习的时候只对比环
            contrast_nodes = ring_nodes

        # # print(h1[mask_nodes].size())
        # # print(h2.size())
        # # print(h1[mask_nodes].mean(dim=0).size())
        # # print(h2.detach().mean(dim=0).size())
        # # print(h1[central_nodes].shape)
        # # print(h1)
        # loss = self.criterion(h1[mask_nodes].mean(dim=0), h2.detach().mean(dim=0))
        # print('mask_nodes:',mask_nodes)
        # print('h1[mask_nodes]:',h1[mask_nodes])
        # print('h1[mask_nodes].shape:',h1[mask_nodes].shape)
        # print('h2:',h2.shape)
        # print('h2.detach():',h2.detach())
        loss = self.criterion(h1[contrast_nodes], h2[contrast_nodes].detach())
        # loss = self.criterion(h1[mask_nodes], h2.detach())
        return loss

    def get_embed(self, graph, feat):
        _, h = self.online_encoder(graph, feat)

        return h.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
