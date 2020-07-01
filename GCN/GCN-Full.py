import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import citation_graph as citegrh
from dgl.data import RedditDataset
import pandas as pd
import time
import datetime 

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None, test=False, concat=False):
        super(NodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        self.concat = concat
        self.test = test

    def forward(self, node):
        h = node.data['h']
        if self.test:
            h = h * node.data['norm']
        h = self.linear(h)
        # skip connection
        if self.concat:
            h = torch.cat((h, self.activation(h)), dim=1)
        elif self.activation:
            h = self.activation(h)
        return {'activation': h}


class GCNSampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCNSampling, self).__init__()
        self.n_layers = n_layers
        if dropout != 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2*n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(dim=1)},
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


class GCNInfer(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation):
        super(GCNInfer, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        skip_start = (0 == n_layers-1)
        self.layers.append(NodeUpdate(in_feats, n_hidden, activation, test=True, concat=skip_start))
        # hidden layers
        for i in range(1, n_layers):
            skip_start = (i == n_layers-1)
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation, test=True, concat=skip_start))
        # output layer
        self.layers.append(NodeUpdate(2*n_hidden, n_classes, test=True))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']

        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            nf.layers[i].data['h'] = h
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'),
                             layer)

        h = nf.layers[-1].data.pop('activation')
        return h


def main(args):
    # load and preprocess dataset
    # data = load_data(args)concole
    # data = RedditDataset(self_loop=True)
    out = [0]
    data = citegrh.load_pubmed()
    node_list = list(range(19717)) 
    list_test = node_list[0::1]

    if args.self_loop and not args.dataset.startswith('reddit'):
        data.graph.add_edges_from([(i,i) for i in range(len(data.graph))])

    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)

    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)

    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()
    n_test_samples_test = n_test_samples
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %dconcole
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           n_train_samples,
    #           n_val_samples,
    #           n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    norm = 1. / g.in_degrees().float().unsqueeze(1)
    features_test = features[list_test]

    num_neighbors = args.num_neighbors
    # get the subgraph for inference
    g_test = g.subgraph(list_test)  
    g_test.copy_from_parent()  
    g_test.readonly()
    labels_test = labels[list_test]
    test_nid_test = []
    norm_test = 1. / g_test.in_degrees().float().unsqueeze(1)

    for i in range(len(list_test)):
        if list_test[i] in test_nid:
            test_nid_test.append(i)
    test_nid_test = np.array(test_nid_test)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        labels_test = labels_test.cuda()

        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

        norm = norm.cuda()
        norm_test = norm_test.cuda()
        features_test = features_test.cuda()

    g.ndata['features'] = features
    g_test.ndata['features'] = features_test
    

    g.ndata['norm'] = norm
    g_test.ndata['norm'] = norm_test

    model = GCNSampling(in_feats,
                        args.n_hidden,
                        n_classes,
                        args.n_layers,
                        F.relu,
                        args.dropout)

    infer_model = GCNInfer(in_feats,
                    args.n_hidden,
                    n_classes,
                    args.n_layers,
                    F.relu)

    if cuda:
        model.cuda()
        infer_model.cuda()

    loss_fcn = nn.CrossEntropyLoss()



    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    time_now = time.time()
    for epoch in range(args.n_epochs):
        
        for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
            # print(nf.layer_size(0))
            nf.copy_from_parent()
            model.train()
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            loss = loss_fcn(pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_next = time.time()
        # # print('cost: ',round((time_next-time_now),4))
        # time_now = time_next            # print(nf.layer_size(0))
        # for infer_param, param in zip(infer_model.parameters(), model.parameters()):
        #     infer_param.data.copy_(param.data)

        num_acc = 0.

        for infer_param, param in zip(infer_model.parameters(), model.parameters()):  
            infer_param.data.copy_(param.data)
        for nf in dgl.contrib.sampling.NeighborSampler(g_test, args.test_batch_size,
                                                       g_test.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid_test):
            nf.copy_from_parent()
            infer_model.eval()
            with torch.no_grad():
                pred = infer_model(nf)
                batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
                batch_labels = labels_test[batch_nids]
                num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
        acc = round(num_acc/n_test_samples_test,4)
        print('In round: ',epoch,' Test Accuracy: ',acc,'Test Loss: ',loss.data,' Test cost: ', round((time_next-time_now),4))
        out.append(acc)
        time_now = time_next
    dataframe = pd.DataFrame({'acc':out})
    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Pubmed/acc_full.csv",header = False,index=False,sep=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.001,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=10000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=20000,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=32,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()


    main(args)

