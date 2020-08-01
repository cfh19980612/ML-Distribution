from __future__ import division
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
import scipy.sparse as sp
from numpy import linalg as la
from scipy.linalg import fractional_matrix_power



acc=0

def pca_svd(data, k):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    # SVD
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])



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
def load_cora_data(Client, list_test, num_clients):
    # data = RedditDataset(self_loop=True)
    data = citegrh.load_citeseer()
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
    n_test_samples_test = n_test_samples


    features_test = features[list_test]
    norm_test = norm[list_test]

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

    # g1 = g.subgraph(list1)  
    # g1.copy_from_parent()  
    # g1.readonly()
    # g2 = g.subgraph(list2)  
    # g2.copy_from_parent()  
    # g2.readonly() 
    # g3 = g.subgraph(list3)  
    # g3.copy_from_parent()  
    # g3.readonly()
    g_test = g.subgraph(list_test)  
    g_test.copy_from_parent()  
    g_test.readonly()
    #g.readonly()

    # labels1 = labels[list1]
    # labels2 = labels[list2]
    # labels3 = labels[list3]
    labels_test = labels[list_test]
    

    train_nid1 = []
    train_nid2 = []
    train_nid3 = []
    train_nid4 = []
    train_nid5 = []
    train_nid6 = []
    train_nid7 = []
    train_nid8 = []
    train_nid9 = []
    train_nid0 = []

    Train_nid = [train_nid1, train_nid2, train_nid3,train_nid4, train_nid5, train_nid6, train_nid7, train_nid8, train_nid9, train_nid0]
    test_nid_test = []

    for i in range(num_clients):
        for j in range(len(Client[i])):
            if Client[i][j] in train_nid:
                Train_nid[i].append(Client[i][j])
        Train_nid[i] = np.array(Train_nid[i])


    # for i in range(len(Client[0])):
    #     if Client[0][i] in train_nid:
    #         train_nid1.append(Client[0][i])
    # train_nid1 = np.array(train_nid1)

    # for i in range(len(list2)):
    #     if list2[i] in train_nid:
    #         train_nid2.append(list2[i])
    # train_nid2 = np.array(train_nid2)

    # for i in range(len(list3)):
    #     if list3[i] in train_nid:
    #         train_nid3.append(list3[i])
    # train_nid3 = np.array(train_nid3)

    for i in range(len(list_test)):
        if list_test[i] in test_nid:
            test_nid_test.append(i)
    test_nid_test = np.array(test_nid_test)

    return g, g_test,norm_test,features_test,train_mask,test_mask,labels, labels_test, train_nid, Train_nid, test_nid, test_nid_test, in_feats, n_classes, n_test_samples, n_test_samples_test

# run a subgraph
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,train_prob):
    loss_fcn = nn.CrossEntropyLoss()
    if cuda == True:
        Model.cuda()
    # sampling
    # time_now = time.time()
    time_cost = 0
    for nf in dgl.contrib.sampling.LayerSampler(Graph, args.batch_size,  
                                                            layer_sizes=[64,64],
                                                            node_prob=train_prob,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=32,
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
    p = Model.state_dict()
    if cuda == True:
        Model.cpu()
    # print ('time: ',time_cost)
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

def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda,test_prob):

    num_acc = 0.
    for nf in dgl.contrib.sampling.LayerSampler(g, args.test_batch_size,
                                                       layer_sizes=[20000,20000],
                                                        node_prob=test_prob,
                                                        neighbor_type='in',
                                                        shuffle=True,
                                                        num_workers=32,
                                                       seed_nodes=test_nid):
        nf.copy_from_parent()
        infer_model.eval()
        with torch.no_grad():
            pred = infer_model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = Labels[batch_nids]
            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()
    acc = round(num_acc/N_test_samples,4)
    return acc


def Gen_args(num):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
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


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    #target
    cora = 0.83
    citeseer = 0.72
    pubmed = 0.82
    reddit = 0.97

    # GPU info
    # pynvml.nvmlInit()

    args = Gen_args(10)   # return the parameters
    num_clients = 8

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
    node_list = list(range(3327))
    Client = [None for i in range (num_clients)]
    for i in range(num_clients):
        Client[i] = node_list[i::num_clients]
    list_test = node_list[0::1]

    # Model and Opt list
    Model = [None for i in range (num_clients)]
    Optimizer = [None for i in range (num_clients)]

    # Input Example
    train_prob = np.array([])
    test_prob = np.array([])
    

    # GCN parameter
    g, g_test,norm_test,features_test,train_mask,test_mask, \
        labels, labels_test, train_nid, Train_nid, test_nid, test_nid_test, \
            in_feats, n_classes, n_test_samples, n_test_samples_test = load_cora_data(Client, list_test, num_clients)
    infer_model = genGraph(args,in_feats,n_classes,2)

    # normal Laplacian
    degree = g.out_degrees().numpy()
    D = np.mat(np.zeros((g.number_of_nodes(),g.number_of_nodes())))
    # D = sp.coo_matrix((Degree,(g.number_of_nodes(),g.number_of_nodes())))
    for i in range (g.number_of_nodes()):
        D[i,i] = degree[i]
    A = g.adjacency_matrix(True).to_dense().numpy()
    D = fractional_matrix_power(D,-0.5)
    P = np.dot(D,A,D)

    # calculate the probability for each node
    norm_degree = np.linalg.norm(P,axis=0,keepdims=True).flatten()
    all_degree = 0
    train_probs = np.array([])
    test_probs = np.array([])
    for i in range (g.number_of_nodes()):
        all_degree += norm_degree[i]
    for i in range (g.number_of_nodes()):
        train_probs = np.append(train_probs, round(norm_degree[i]/all_degree,6))
    test_probs = [1.000000 for i in range (g.number_of_nodes())]

    # for i in range (g.number_of_nodes()):
    #     train_probs.append()

    # gpu
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        # for i in range(num_clients):
        #     Model[i].cuda()
        infer_model.cuda()

        labels.cuda()
        # labels1.cuda()
        # labels2.cuda()
        # labels3.cuda()
        labels_test.cuda()

    # initialize model and Opt
    for i in range(num_clients):
        Model[i], Optimizer[i] = genGraph(args,in_feats,n_classes,1)

    # Valueheterogeneous
    for nodes in range(g.number_of_nodes()):
        train_prob = np.append(train_prob, 0.000500)
    for nodes in range(g.number_of_nodes()):
        test_prob = np.append(test_prob, 1)

    s = []
    s_ = []
    P = [None for i in range (num_clients)]
    Time_cost = [None for i in range (num_clients)]
    Loss = [None for i in range (num_clients)]
    for epoch in range(args.n_epochs):
        for i in range(num_clients):
            P[i], Time_cost[i], Loss[i] = runGraph(Model[i],g,args,Optimizer[i],labels,Train_nid[i],cuda,train_probs)
        
        # loss
        # loss = 0
        # for i in num_clients:
        #     loss += Loss[i]
        # loss = round(loss/3,4)

        # time cost
        time_cost = 0
        for i in range(num_clients):
            time_cost += Time_cost[i]
        time_cost = round(time_cost/num_clients,4)

        # aggregation
        for key, value in P[0].items():  
            for i in range(num_clients):
                if i == 0:
                    P[0][key] = P[i][key] * (len(Train_nid[i]) / len(train_nid))
                else:
                    P[0][key] += P[i][key] * (len(Train_nid[i]) / len(train_nid))

        for i in range(num_clients):
            Model[i].load_state_dict(P[0])
        
        for infer_param, param in zip(infer_model.parameters(), Model[0].parameters()):  
            infer_param.data.copy_(param.data)
        
        # test 
        acc = inference(g,infer_model,args,labels,test_nid,in_feats,n_classes,n_test_samples,cuda,test_prob)

        if epoch > 0:
            times = times + time_cost
            X.append(times)
            Y.append(acc)
            print('Epoch: ',epoch,'||', 'Accuracy: ', acc, '||', 'Timecost: ', times)
        if acc >= citeseer:
            break
    

        

        
    dataframes = pd.DataFrame(Y, columns=['Y'])
    dataframe = pd.DataFrame(X, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(Y,columns=['Y'])],axis=1)
    
    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/acc_gcn_FastGCN.csv",header = False,index=False,sep=',')
    dataframes.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/acc_gcn_FastGCN(round).csv",header = False,index=False,sep=',')
        