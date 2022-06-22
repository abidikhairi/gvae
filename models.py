import math
import torch as th
from torch import nn
from torch.nn import functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_feats, out_feats) -> None:
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.weight = nn.Parameter(th.FloatTensor(in_feats, out_feats))
        self.bias = nn.Parameter(th.FloatTensor(out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = th.mm(x, self.weight)
        output = th.mm(adj, support)
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


class GAE(nn.Module):
    def __init__(self, feature_size, hidden_size, code_size) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.code_size = code_size

        self.gcn = GCN(feature_size, hidden_size, code_size)


    def encoder(self, x, adj):
        return self.gcn(x, adj)


    def decoder(self, z):    
        return th.sigmoid(th.matmul(z, z.t()))


    def forward(self, x, adj):
        z = self.encoder(x, adj)
        adj_rec = self.decoder(z)

        return z, adj_rec


class VGAE(nn.Module):
    def __init__(self, feature_size, hidden_size, code_size) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.code_size = code_size

        self.gcn = GCN(feature_size, hidden_size, hidden_size)

        self.conv_mean = GraphConvolutionLayer(hidden_size, code_size)
        self.conv_log_std = GraphConvolutionLayer(hidden_size, code_size)


    def encoder(self, x, adj):
        hn = self.gcn(x, adj)
        self.mean = self.conv_mean(hn, adj)
        self.log_std = self.conv_log_std(hn, adj)
        noise = th.randn(x.size(0), self.code_size).to(x.device)
        
        z = self.mean + noise * th.exp(self.log_std)
        return z


    def decoder(self, z):
        adj_rec = th.sigmoid(th.matmul(z, z.t()))
        return adj_rec


    def forward(self, x, adj):
        z = self.encoder(x, adj)
        adj_rec = self.decoder(z)

        return z, adj_rec
