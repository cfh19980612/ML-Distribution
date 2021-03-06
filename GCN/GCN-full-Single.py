import argparse, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import citation_graph as citegrh
from dgl.data import RedditDataset
import pandas as pd
import networkx as nx
#import thread
import threading
# import multiprocessing as mp
import psutil
import os
import time 
import datetime
import math

acc=0


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

# create the subgraph
def load_cora_data():
    # data = RedditDataset(self_loop=True)
    data = citegrh.load_pubmed()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)
    g = data.graph
    # add self loop

    g = DGLGraph(data.graph, readonly=True)
    n_classes = data.num_labels


    norm = 1. / g.in_degrees().float().unsqueeze(1)
    in_feats = features.shape[1]
    n_test_samples = test_mask.int().sum().item()




    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        #val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        norm = norm.cuda()

    g.ndata['features'] = features
    # num_neighbors = args.num_neighbors
    g.ndata['norm'] = norm

    return g,train_mask,test_mask,labels, train_nid,  test_nid,  in_feats, n_classes, n_test_samples

# run a subgraph
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,sampling):
    loss_fcn = nn.CrossEntropyLoss()

        # sampling
    # time_now = time.time()
    time_cost = 0
    if cuda:
        Model.cuda()
        Labels.cuda()
        
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            expand_factor = sampling,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=16,
                                                            num_hops=args.n_layers+1,
                                                            seed_nodes=train_nid):
        nf.copy_from_parent()
        time_now = time.time()
        Model.train()

        # forward
        pred = Model(nf)
        batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)  
        batch_labels = Labels[batch_nids]
        loss = loss_fcn(pred, batch_labels)
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        time_next = time.time()
        time_cost += round(time_next-time_now,4)
    # time_cost = round(time_next-time_now,4)
    p = Model.state_dict()
    if cuda:
        Model.cpu()
        Labels.cpu()
        torch.cuda.empty_cache()
    return p, time_cost, loss.data

# generate the subgraph's model and optimizer
def genGraph(args,In_feats,N_classes,flag):
    if flag == 1:
        model = GCNSampling(In_feats,
                            args.n_hidden,
                            N_classes,
                            args.n_layers,
                            F.relu,
                            args.dropout)

        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
        return model, optimizer
    else:   
        infer_model = GCNInfer(In_feats,
                        args.n_hidden,
                        N_classes,
                        args.n_layers,
                        F.relu)
        return infer_model

def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda,sampling):

    num_acc = 0.
    if cuda:
        Labels.cuda()
        infer_model.cuda()
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.test_batch_size,
                                                        expand_factor=sampling,
                                                        neighbor_type='in',
                                                        num_workers=32,
                                                        num_hops=args.n_layers+1,
                                                        seed_nodes=Test_nid):
        nf.copy_from_parent()
        infer_model.eval()
        with torch.no_grad():
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = Labels[batch_nids]
            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
    acc = round(num_acc/N_test_samples,4)
    if cuda:
        infer_model.cpu()
        Labels.cpu()
        torch.cuda.empty_cache()
    return acc


def Gen_args(num):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=1000,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=5000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=num,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
            help="number of hidden gcn layers")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = Gen_args(10)   # return the parameters
    num_clients = 8
    #target
    cora = 0.83
    citeseer = 0.72
    pubmed = 0.8

    # DQN parameter
    A = 0.6
    X = []
    Y = []
    times = 0
    acc_now = 0
    acc_next = 0 
    step = 0
    time_cost_past = 5

    # Client graph list and test
    node_list = list(range(19717))

    # Model and Opt list


    # GCN parameter
    g,train_mask,test_mask, labels, train_nid, test_nid, in_feats, n_classes, n_test_samples = load_cora_data()

    # initialize model and Opt

    Model, Optimizer = genGraph(args,in_feats,n_classes,1)
    infer_model = genGraph(args,in_feats,n_classes,2)

    # gpu
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
    
    # sampling list
    layer_scale = np.array([1,1])

    Batch_sampling_method = []

    test_batch_sampling_method = np.array([])

    #test sampling
    for layer in range(args.n_layers + 1):
        for nodes in range(g.number_of_nodes()):
            test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)

    # train sampling
    for layer in range(args.n_layers + 1):
        for nodes in range(g.number_of_nodes()):
            temp = math.ceil(g.in_degree(nodes) * layer_scale[layer])
            Batch_sampling_method = np.append(Batch_sampling_method, temp)
    print (Batch_sampling_method)


    for epoch in range(args.n_epochs):
        P, Time_cost, Loss = runGraph(Model,g,args,Optimizer,labels,train_nid,cuda,Batch_sampling_method)
        
        # loss
        # loss = 0
        # for i in num_clients:
        #     loss += Loss[i]
        # loss = round(loss/3,4)

        # time cost
        time_cost = Time_cost


        
        for infer_param, param in zip(infer_model.parameters(), Model.parameters()):  
            infer_param.data.copy_(param.data)
        
        # test 
        acc = inference(g,infer_model,args,labels,test_nid,in_feats,n_classes,n_test_samples,cuda,test_batch_sampling_method)

        if epoch > 0:
            times = times + time_cost
            X.append(times)
            Y.append(acc)
            print('Epoch: ',epoch,'||', 'Accuracy: ', acc, '||', 'Timecost: ', times)
        
        if acc >= pubmed:
            break
        

        
    dataframes = pd.DataFrame(Y, columns=['Y'])
    dataframe = pd.DataFrame(X, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(Y,columns=['Y'])],axis=1)
    
    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Pubmed(8)/single/acc_gcn_nonsampling.csv",header = False,index=False,sep=',')
    dataframes.to_csv("/home/fahao/Py_code/results/GCN-Pubmed(8)/single/acc_gcn_nonsampling(round).csv",header = False,index=False,sep=',')

        