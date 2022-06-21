import torch as th


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


def split_edges(edges: th.Tensor):
    """
    Split the edges into three parts.
        10% of the edges are used for validation
        20% of the edges are used for testing
        70% of the edges are used for training
    """
    num_test = int(edges.shape[1] * 0.2)
    num_val = int(edges.shape[1] * 0.1)
    
    edge_random_idx = th.randperm(edges.shape[1]).to(edges.device)

    test_edges = th.index_select(edges, 1, edge_random_idx[:num_test])
    val_edges = th.index_select(edges, 1, edge_random_idx[num_test:num_test+num_val])
    train_edges = th.index_select(edges, 1, edge_random_idx[num_test+num_val:])

    return train_edges, val_edges, test_edges
