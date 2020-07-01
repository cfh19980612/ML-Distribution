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
def load_cora_data(list1, list2, list3, list_test):
    data = RedditDataset(self_loop=True)
    # data = citegrh.load_citeseer()
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
    n_test_samples_test = n_test_samples/4

    features1 = features[list1]
    norm1 = norm[list1]

    features2 = features[list2]
    norm2 = norm[list2]

    features3 = features[list3]
    norm3 = norm[list3]

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

    g1 = g.subgraph(list1)  
    g1.copy_from_parent()  
    g1.readonly()
    g2 = g.subgraph(list2)  
    g2.copy_from_parent()  
    g2.readonly() 
    g3 = g.subgraph(list3)  
    g3.copy_from_parent()  
    g3.readonly()
    g_test = g.subgraph(list_test)  
    g_test.copy_from_parent()  
    g_test.readonly()
    #g.readonly()

    labels1 = labels[list1]
    labels2 = labels[list2]
    labels3 = labels[list3]
    labels_test = labels[list_test]

    train_nid1 = []
    train_nid2 = []
    train_nid3 = []
    test_nid_test = []

    for i in range(len(list1)):
        if list1[i] in train_nid:
            train_nid1.append(i)
    train_nid1 = np.array(train_nid1)

    for i in range(len(list2)):
        if list2[i] in train_nid:
            train_nid2.append(i)
    train_nid2 = np.array(train_nid2)

    for i in range(len(list3)):
        if list3[i] in train_nid:
            train_nid3.append(i)
    train_nid3 = np.array(train_nid3)

    for i in range(len(list_test)):
        if list_test[i] in test_nid:
            test_nid_test.append(i)
    test_nid_test = np.array(test_nid_test)

    return g, g1, g2, g3, g_test, norm1,norm2,norm3,norm_test,features1,features2,features3,features_test,train_mask,test_mask,labels, labels1, labels2, labels3, labels_test, train_nid, train_nid1, train_nid2,train_nid3, test_nid, test_nid_test, in_feats, n_classes, n_test_samples, n_test_samples_test



# run a subgraph
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,num_neighbors):
    loss_fcn = nn.CrossEntropyLoss()

        # sampling
    time_now = time.time()
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            expand_factor = args.num_neighbors,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=10,
                                                            num_hops=args.n_layers+1,
                                                            seed_nodes=train_nid):
        nf.copy_from_parent()
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
    time_cost = round(time_next-time_now,4)
    p = Model.state_dict()

    return p, time_cost, loss.data





# generate the subgraph's model and optimizer
def genGraph(args,In_feats,N_classes,N_test_samples,flag):
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

def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda):

    num_acc = 0.

    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.test_batch_size,
                                                        Graph.number_of_nodes(),
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


        # r = pow(64,(acc-A))+math.log(time_cost_past-time_cost)
        # s_[0] = acc
        # s_ = np.array(s_)
        # RL.store_transition(s, num_neighbors-1, r, s_)
        # if (step > 30) and (step % 5 == 0):
        #     RL.learn()
        # s = s_
        # time_cost_past = time_cost
        # step += 1
        # acc_now = acc
    print('In round: ',epoch,' The Accuracy: ',acc)
    return acc


def Gen_args(num):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.7,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.0001,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=num,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=128,
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
    
    # DQN parameter
    A = 0.6
    out=[0]
    acc_now = 0
    acc_next = 0 
    step = 0
    time_cost_past = 5

    # pubmed 
    node_list = list(range(232965))  
    list1 = node_list[0::3]
    list2 = node_list[1::3]
    list3 = node_list[2::3]
    list_test = node_list[0::4]

    # GCN parameter
    g, g1, g2, g3, g_test,norm1,norm2,norm3,norm_test,features1,features2,features3,features_test,train_mask,test_mask, labels, labels1, labels2, labels3, labels_test, train_nid, train_nid1, train_nid2, train_nid3, test_nid, test_nid_test, in_feats, n_classes, n_test_samples, n_test_samples_test = load_cora_data(list1, list2, list3, list_test)
    
    model1, optimizer1 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    model2, optimizer2 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    model3, optimizer3 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    infer_model = genGraph(args,in_feats,n_classes,n_test_samples,2)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        model1.cuda()
        model2.cuda()
        model3.cuda()
        infer_model.cuda()

        labels.cuda()
        labels1.cuda()
        labels2.cuda()
        labels3.cuda()
        labels_test.cuda()

    s = []
    s_ = []
    for epoch in range(args.n_epochs):

        time_now = time.time()
        p1, time_cost1, loss1 = runGraph(model1,g1,args,optimizer1,labels1,train_nid1,cuda,args.num_neighbors)
        p2, time_cost2, loss2 = runGraph(model2,g2,args,optimizer2,labels2,train_nid2,cuda,args.num_neighbors)
        p3, time_cost3, loss3 = runGraph(model3,g3,args,optimizer3,labels3,train_nid3,cuda,args.num_neighbors)
        
        # loss
        loss = (loss1 + loss2 + loss3)/3

        # time cost
        time_cost = round((time_cost1+time_cost2+time_cost3)/3,4)

        # aggregation
        for key, value in p2.items():  
            p1[key] = p1[key] * (len(train_nid1) / len(train_nid)) + p2[key] * (len(train_nid2) / len(train_nid)) + p3[key] * (len(train_nid3) / len(train_nid))

        model1.load_state_dict(p1)
        model2.load_state_dict(p1)
        model3.load_state_dict(p1)

        for infer_param, param in zip(infer_model.parameters(), model1.parameters()):  
            infer_param.data.copy_(param.data)
        
        # test 
        acc = inference(g_test,infer_model,args,labels_test,test_nid_test,in_feats,n_classes,n_test_samples_test,cuda)

        out.append(acc)
        time_end = time.time()
        print(loss,round(time_end-time_now,4))

        # if acc >= 0.72:
        #     print('Training complete in round: ',epoch)
        #     break
    dataframe = pd.DataFrame({'acc':out})
    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Reddit/acc_nondqn_10.csv",header = False,index=False,sep=',')

        

        



    
        # if acc >= 0.72:
        #     print('Training complete in round: ',epoch)
        #     break

