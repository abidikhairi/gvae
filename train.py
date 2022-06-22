import os
import logging
import argparse
import torch as th
from torch.nn import functional as F

from models import GAE
from utils import get_scores, normalize_adjacency, get_edges, split_edges, compute_loss_para


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    
    root = args.root
    dataset = args.dataset
    epochs = args.epochs
    lr = args.lr
    hidden_size = args.hidden_size
    code_size = args.code_size

    device = th.device('cpu' if th.cuda.is_available() else 'cpu')
    
    adj = th.load(os.path.join(root, dataset, 'adj.pt')).to_dense().to(device)
    features = th.load(os.path.join(root, dataset, 'feats.pt'))

    features = features.to(device)

    edges = get_edges(adj)
    train_adj, _, val_edges, val_edges_false, test_edges, test_edges_false = split_edges(edges, adj)
    
    weight_tensor, norm = compute_loss_para(train_adj)
    train_adj_norm = normalize_adjacency(train_adj).to(train_adj.device)
    feature_size = features.shape[1]
    
    gae = GAE(feature_size, hidden_size, code_size)

    gae.to(device)

    optimizer = th.optim.Adam(gae.parameters(), lr=lr)
    
    logger.info('Total Parameters: {}'.format(sum([p.nelement() for p in gae.parameters()])))

    
    for epoch in range(epochs):
        gae.train()
        optimizer.zero_grad()
        
        _, logits = gae(features, train_adj_norm)
        
        reconstruction_loss = norm * F.binary_cross_entropy(logits.view(-1), train_adj.view(-1), weight=weight_tensor)

        reconstruction_loss.backward()
        optimizer.step()

        val_auroc, val_ap = get_scores(val_edges, val_edges_false, logits)

        logger.info('epoch: {:3d} training Loss: {:.4f} validation auroc: {:.2f} % validation ap: {:.2f} % '.format(
            epoch, reconstruction_loss.item(), val_auroc * 100, val_ap * 100)
        )

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    
    logger.info('training finished !!')
    logger.info('test auroc: {:.2f} % test ap: {:.2f} % '.format(test_roc * 100, test_ap * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--hidden-size', type=int, default=32, help='number of hidden units')
    parser.add_argument('--code-size', type=int, default=16, help='number of hidden units')

    args = parser.parse_args()
    main(args)
