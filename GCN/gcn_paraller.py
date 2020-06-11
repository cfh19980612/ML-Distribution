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
import networkx as nx
#import thread
import threading
import multiprocessing as mp

acc=0

class myThread (threading.Thread):
    def __init__(self, threadID, name,num_neighbors,flag):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.num_neighbors = num_neighbors
        self.flag = falg
        #self.counter = counter
    def run(self):
        print ('Starting ' + self.name)
        threadLock.acquire()
        start(self.name,self.num_neighbors,self.flag)
        threadLock.release()

threadLock = threading.Lock()
threads = []

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
def load_cora_data(list1, list2):
    
    data = citegrh.load_pubmed()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    n_classes = data.num_labels
    g.add_edges(g.nodes(), g.nodes())

    norm = 1. / g.in_degrees().float().unsqueeze(1)
    g.ndata['features'] = features
    # num_neighbors = args.num_neighbors
    g.ndata['norm'] = norm

    in_feats = features.shape[1]
    n_test_samples = test_mask.int().sum().item()

    # src, dst = g.all_edges()
    # src = src.detach().numpy()
    # dst = dst.detach().numpy()
    # edge_list = list(zip(src, dst))  

    g1 = g.subgraph(list1)  
    g1.copy_from_parent()  
    g1.readonly()
    g2 = g.subgraph(list2)  
    g2.copy_from_parent()  
    g2.readonly()
    g.readonly()

    labels1 = labels[list1]
    labels2 = labels[list2]

    train_nid1 = []
    train_nid2 = []
    for i in range(len(list1)):
        if list1[i] in train_nid:
            train_nid1.append(i)
    train_nid1 = np.array(train_nid1)

    for i in range(len(list2)):
        if list2[i] in train_nid:
            train_nid2.append(i)
    train_nid2 = np.array(train_nid2)

    return g, g1, g2, labels, labels1, labels2, train_nid, train_nid1, train_nid2, test_nid, in_feats, n_classes, n_test_samples

# run a subgraph
def runGraph(Model,Graph,args,Optimizer,Labels,Train_nid,pipe):
    loss_fcn = nn.CrossEntropyLoss()
    for epoch in range(args.n_epochs):
        for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            args.num_neighbors,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=32,
                                                            num_hops=args.n_layers+1,
                                                            seed_nodes=Train_nid):
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
        p = Model.state_dict()
        pipe.send(p)
        p1 = pipe.recv()
        Model.load_state_dict(p1)
        pipe.send(Model)



# generate the subgraph's model and optimizer
def genGraph(args,Graph,Labels,Train_nid,In_feats,N_classes,N_test_samples):
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

def inference(Graph,args,Labels,Test_nid,Train_nid,Train_nid1,Train_nid2,In_feats,N_classes,N_test_samples,pipe_1,pipe_2):

    for epoch in range(args.n_epochs):
        p1 = pipe_1.recv()
        p2 = pipe_2.recv()
        
        infer_model = GCNInfer(In_feats,
                            args.n_hidden,
                            N_classes,
                            args.n_layers,
                            F.relu)

        for key, value in p2.items():  
                p1[key] = p1[key] * (len(Train_nid1) / len(Train_nid)) + p2[key] * (len(Train_nid2) / len(Train_nid))

            # model1.load_state_dict(p1)  
            # model2.load_state_dict(p1)  
        pipe_1.send(p1)
        pipe_2.send(p1)
        model1 = pipe_1.recv()
        model2 = pipe_2.recv()
        for infer_param, param in zip(infer_model.parameters(), model1.parameters()):  
            infer_param.data.copy_(param.data)

        num_acc = 0.

        for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.test_batch_size,
                                                        g.number_of_nodes(),
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
        acc = round(num_acc/n_test_samples,4)
        print("round: ",epoch," Test Accuracy :", acc)

    # dataframe = pd.DataFrame({'acc':out})
    # dataframe.to_csv("non-dqn-4.csv",header = False,index=False,sep=',')


def Gen_args(num_neighbors):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
            help="test batch size")

    parser.add_argument("--num-neighbors", type=int, default=num_neighbors,
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
    args = Gen_args(5)   # return the parameters

    # communication
    pipe1 = mp.Pipe()
    pipe2 = mp.Pipe()


    out = [0]

    node_list = list(range(19717))  
    list1 = node_list[0::2]
    list2 = list(set(node_list) - set(list1))
    g, g1, g2, labels, labels1, labels2, train_nid, train_nid1, train_nid2, test_nid, in_feats, n_classes, n_test_samples = load_cora_data(list1, list2)
    
    model1, optimizer1 = genGraph(args,g1,labels1,train_nid1,in_feats,n_classes,n_test_samples)
    model2, optimizer2 = genGraph(args,g2,labels2,train_nid2,in_feats,n_classes,n_test_samples)

    task1 = mp.Process(target=runGraph,args=(model1,g1,args,optimizer1,labels1,train_nid1,pipe1[0],))
    task2 = mp.Process(target=runGraph,args=(model2,g2,args,optimizer2,labels2,train_nid1,pipe2[0],))
    task3 = mp.Process(target=inference,args=(g,args,labels,test_nid,train_nid,train_nid1,train_nid2,in_feats,n_classes,n_test_samples,pipe1[1],pipe2[1],))

    task1.start()
    task2.start()
    task3.start()

    task1.join()
    task2.join()
    task3.join()


