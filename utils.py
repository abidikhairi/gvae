import torch as th
import numpy as np
import torchmetrics.functional as thm


def normalize_adjacency(adj):
    """
    Normalize the adjacency matrix.
    """
    adj = adj + th.eye(adj.shape[0]).to(adj.device)
    deg = th.sum(adj, dim=0)
    deg_inv_sqrt = th.pow(deg, -0.5)
    deg_inv_sqrt = th.diag(deg_inv_sqrt)

    return th.mm(th.mm(deg_inv_sqrt, adj), deg_inv_sqrt)


def get_edges(adj: th.Tensor):
    """
    Get the edges of the adjacency matrix.
    """
    
    adj.diag().fill_(0)
    
    return th.nonzero(adj, as_tuple=False).t()


def split_edges(edges: th.Tensor, adj: th.Tensor):
    """
    Split the edges into three parts.
        10% of the edges are used for validation
        20% of the edges are used for testing
        70% of the edges are used for training
    """

    def ismemeber(a, b):
        a, b = np.array(a), b.t().numpy()
        rows_close = np.all(np.round(a - b[:, None], 5) == 0, axis=-1)
        return np.any(rows_close)
    
    num_nodes = adj.shape[0]
    num_test = int(edges.shape[1] * 0.2)
    num_val = int(edges.shape[1] * 0.1)
    
    edge_random_idx = th.randperm(edges.shape[1]).to(edges.device)

    test_edges = th.index_select(edges, 1, edge_random_idx[:num_test])
    val_edges = th.index_select(edges, 1, edge_random_idx[num_test:num_test+num_val])
    train_edges = th.index_select(edges, 1, edge_random_idx[num_test+num_val:])

    val_edges_false = []

    while len(val_edges_false) < val_edges.shape[1]:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)

        if src == dst:
            continue
    
        if ismemeber([src, dst], edges):
            continue
        
        if val_edges_false:
            if ismemeber([src, dst], edges):
                continue
            if ismemeber([dst, src], edges):
                continue

        val_edges_false.append([src, dst])
        

    test_edge_false = []

    while len(test_edge_false) < test_edges.shape[1]:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)

        if src == dst:
            continue
    
        if ismemeber([src, dst], edges):
            continue
        
        if test_edge_false:
            if ismemeber([src, dst], edges):
                continue
            if ismemeber([dst, src], edges):
                continue
        
        test_edge_false.append([src, dst])

    val_edges_false = th.tensor(val_edges_false).t().to(edges.device)
    test_edge_false = th.tensor(test_edge_false).t().to(edges.device)
    
    values = th.ones(train_edges.shape[1])
    train_adj = th.sparse_coo_tensor(indices=train_edges, values=values, size=(num_nodes, num_nodes)).to_dense().to(adj.device)
    train_adj = train_adj

    return train_adj, train_edges, val_edges, val_edges_false, test_edges, test_edge_false


def compute_loss_para(adj):
    
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = th.ones(weight_mask.size(0)).to(adj.device)
    weight_tensor[weight_mask] = pos_weight
    
    return weight_tensor, norm


def get_scores(pos_edges, neg_edges, adj):
    adj = adj.detach().cpu()
    
    preds = []
    neg_preds = []
    for edge in pos_edges.t():
        preds.append(th.sigmoid(adj[edge[0], edge[1]]).item())
    
    for edge in neg_edges.t():
        neg_preds.append(th.sigmoid(adj[edge[0], edge[1]]).item())

    preds_all = th.hstack([
        th.tensor(preds),
        th.tensor(neg_preds)
    ])

    labels_all = th.hstack([
        th.ones(len(preds)),
        th.zeros(len(neg_preds))
    ]).long()


    auroc_score = thm.auroc(preds_all, labels_all).item()
    ap_score = thm.average_precision(preds_all, labels_all).item()

    return auroc_score, ap_score
