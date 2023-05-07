from collections import namedtuple, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import TUDataset
import dgl
import networkx as nx
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
def pyg2dgl(pyg_data):
    # print(pyg_data.x)
    # 将x转化为torch.float32
    # print(pyg_data.x)
    # print(pyg_data.edge_index)
    # 转换节点特征
    
    node_feat = torch.Tensor(pyg_data.x.float())
    # node_label = torch.Tensor(pyg_data.y.float())

    # label = torch.Tensor(pyg_data.y)
    # 转换边特征
    # edge_feat = torch.Tensor(pyg_data.edge_attr)
    # edge_feat = torch.ones(pyg_data.edge_index.shape[1], 1) # 无边特征时，初始化一个全1特征

    # 空的DGL对象
    g = dgl.DGLGraph()

    # 添加节点
    g.add_nodes(pyg_data.num_nodes)
    # 添加边
    src, dst = pyg_data.edge_index
    g.add_edges(src, dst)
    # 设置节点和边特征
    g.ndata['feat'] = node_feat
    # print(g)

    # 持久化，获取的dgl_data和g是一样的
    ## save_graphs('./data/dgl_data.bin', g)
    ## print(type(g))
    ## g_list, _ = dgl.load_graphs('./data/dgl_data.bin')
    ## dgl_data = g_list[0]
    ## print(type(dgl_data))
    assert type(g) == dgl.DGLGraph
    return g

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask


def graph_show(graph, index):
    sns.set_palette(sns.color_palette("pastel"))
    g = nx.Graph(dgl.to_networkx(graph))
    nx.draw(g, pos=nx.kamada_kawai_layout(g), node_color=graph.ndata["node_labels"].numpy(), node_size=200, width=3)
    # nx.draw(g, pos=nx.spring_layout(g), node_color=graph.ndata["node_labels"].numpy(), node_size=50)
    plt.savefig(f'./PROTEINS/{index}.png', dpi=800)
    # plt.show()
    plt.close()


def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    # if ("attr" not in graph.ndata) and ("node_attr" not in graph.ndata):
    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())

            feature_dim += 1
            # i = 0
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                # print(node_label)
                # graph_show(g, i)
                # i += 1
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES

                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use attr as node features ********")
        # for g, l in dataset:
        #     g.ndata['attr'] = g.ndata['node_attr'].float()
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])

    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]
    Y = np.array([y.numpy()[0] for _, y in dataset])
    # print(f"******** # Num Graphs: {len(dataset1)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    # mask = train_test_split(
    #     Y, seed=np.random.randint(0, 35456, size=1), train_examples_per_class=400,
    #     val_size=0, test_size=None)
    # dataset1 = [(dataset[index]) for index, i in enumerate(mask['train'].astype(bool)) if i == True]

    # return dataset, (feature_dim, num_classes), mask['train'].astype(bool), mask['test'].astype(bool)
    return dataset, (feature_dim, num_classes)
