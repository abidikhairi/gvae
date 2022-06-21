import torch as th
from torch import nn
from torch.nn import functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.Tensor(out_feats))

    def reset_parameters(self):
        stdv = 1. / th.sqrt(self.weight.size(self.out_feats))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = th.mm(adj, x)
        output = th.mm(support, self.weight)
        output = output + self.bias

        return output


class GCN(nn.Module):
    def __init__(self, nfeats, nhids, nclasses, dropout=0.5):
        super().__init__()
        
        self.dropout = dropout

        self.conv1 = GraphConvolutionLayer(nfeats, nhids)
        self.conv2 = GraphConvolutionLayer(nhids, nclasses)
    
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)

        return x
