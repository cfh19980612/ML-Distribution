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
from dqn_agent_torch import DQN
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
    # data = citegrh.load_pubmed()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    train_nid = np.nonzero(data.train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(data.test_mask)[0].astype(np.int64)
    g = data.graph
    # add self loop
    #g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(data.graph, readonly=True)
    n_classes = data.num_labels
    # g.add_edges(g.nodes(), g.nodes())

    norm = 1. / g.in_degrees().float().unsqueeze(1)
    in_feats = features.shape[1]
    n_test_samples = test_mask.int().sum().item()
    n_test_samples_test = n_test_samples/10

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

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     labels1 = labels.cuda()
    #     labels2 = labels.cuda()
    #     labels3 = labels.cuda()
    #     labels_test = labels.cuda()
        #val_mask = val_mask.cuda()
    

    return g, g1, g2, g3, g_test, norm1,norm2,norm3,norm_test,features1,features2,features3,features_test,train_mask,test_mask,labels, labels1, labels2, labels3, labels_test, train_nid, train_nid1, train_nid2,train_nid3, test_nid, test_nid_test, in_feats, n_classes, n_test_samples, n_test_samples_test



# run a subgraph
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,num_neighbors):
    loss_fcn = nn.CrossEntropyLoss()

    # bool GPU 
    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     torch.cuda.set_device(args.gpu)
    #     features = features.cuda()
    #     Labels = Labels.cuda()
    #     train_mask = train_mask.cuda()
    #     # #val_mask = val_mask.cuda()
    #     # test_mask = test_mask.cuda()
    #     norm = norm.cuda()
    #     Model.cuda()
    # Graph.ndata['features'] = features
    # Graph.ndata['norm'] = norm

    # start training
    # for epoch in range(args.n_epochs):

        # get the num_neighbor form agent
        # num_neigh = pipe.recv()

        # sampling
    time_now = time.time()
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            expand_factor = num_neighbors,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=32,
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
    # print(loss)
    return p, time_cost
        # pipe.send(p)
        # p1 = pipe.recv()
        # Model.load_state_dict(p1)
        # pipe.send(Model)

        # # q.put(p)
        # # p1 = q.get()
        # # q.put(Model)

        # # round complete
        # pipe.recv()




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

    
    # if args.gpu == 0:
    #     model.cuda()

def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda):
    # A = 0.9
    # out=[0]
    # acc_now = 0
    # acc_next = 0 
    # step = 0
    # time_cost_past = 5
    # num_neighbors = args.num_neighbors

    # infer_model = GCNInfer(In_feats,
    #                 args.n_hidden,
    #                 N_classes,
    #                 args.n_layers,
    #                 F.relu)

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     torch.cuda.set_device(args.gpu)
    #     features = features.cuda()
    #     Labels = Labels.cuda()
    #     # train_mask = train_mask.cuda()
    #     #val_mask = val_mask.cuda()
    #     test_mask = test_mask.cuda()
    #     norm = norm.cuda()
    #     infer_model.cuda()
    
    # Graph.ndata['features'] = features
    # Graph.ndata['norm'] = norm

    #RL = DeepQNetwork(10,1, output_graph=False)
    # for epoch in range(args.n_epochs):
        # if epoch != 0:
        #     s = [0]
        #     s[0] = acc_now
        #     s = np.array(s)
        #     # s[0] = acc_now
        #     # s = np.array(s)
        #     num_neighbors = RL.choose_action(s)+1
        # pipe_1.send(num_neighbors)
        # pipe_2.send(num_neighbors)
        # pipe_3.send(num_neighbors)    

        # time_now = time.time()
        # p1 = pipe_1.recv()
        # p2 = pipe_2.recv()
        # p3 = pipe_3.recv()
        # time_next = time.time()

        # training complete
        # time_cost = round(time_next-time_now,4)

        # for key, value in p2.items():  
        #         p1[key] = p1[key] * (len(Train_nid1) / len(Train_nid)) + p2[key] * (len(Train_nid2) / len(Train_nid)) + p3[key] * (len(Train_nid3) / len(Train_nid))

        # send the new paremeter
        # pipe_1.send(p1)
        # pipe_2.send(p1)
        # pipe_3.send(p1)
        # q1.put(p1)
        # q2.put(p1)
        # q2.put(p1)


        # model1 = pipe_1.recv()
        # model2 = pipe_2.recv()
        # model3 = pipe_3.recv()

        # pipe_1.send('completed')
        # pipe_2.send('completed')
        # pipe_3.send('completed')

        # for infer_param, param in zip(infer_model.parameters(), model1.parameters()):  
        #     infer_param.data.copy_(param.data)

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

    # dataframe = pd.DataFrame({'acc':out})
    # dataframe.to_csv("acc_pubmed_1.csv",header = False,index=False,sep=',')


def Gen_args(num):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.6,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.0003,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=300,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=300,
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
    args = Gen_args(5)   # return the parameters
    
    # DQN parameter
    A = 0.6
    out=[0]
    acc_now = 0
    acc_next = 0 
    step = 0
    time_cost_past = 5

    # communication
    # pipe1 = mp.Pipe()
    # pipe2 = mp.Pipe()
    # pipe3 = mp.Pipe()
    # q1 = mp.Queue()
    # q2 = mp.Queue()
    # q3 = mp.Queue()

    RL = DQN()

    # redditDateset
    # node_list = list(range(232965))  
    # list1 = node_list[0:70000:1]
    # list2 = node_list[70001:140000:1]
    # list3 = node_list[140001:232965:1]
    # # list2 = list(set(node_list) - set(list1))
    # list_test = node_list[0::8]

    # pubmed 
    node_list = list(range(19717))  
    list1 = node_list[0::3]
    list2 = node_list[1::3]
    list3 = node_list[2::3]
    list_test = node_list[0::10]

    # GCN parameter
    g, g1, g2, g3, g_test,norm1,norm2,norm3,norm_test,features1,features2,features3,features_test,train_mask,test_mask, labels, labels1, labels2, labels3, labels_test, train_nid, train_nid1, train_nid2, train_nid3, test_nid, test_nid_test, in_feats, n_classes, n_test_samples, n_test_samples_test = load_cora_data(list1, list2, list3, list_test)
    
    model1, optimizer1 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    model2, optimizer2 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    model3, optimizer3 = genGraph(args,in_feats,n_classes,n_test_samples,1)
    infer_model = genGraph(args,in_feats,n_classes,n_test_samples,2)

    # p1 = model1.state_dict()
    # p2 = model2.state_dict()
    # p3 = model3.state_dict()
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
    time_now = time.time()
    for epoch in range(args.n_epochs):

        # if epoch == 0:
            
        #     num_neighbor = args.num_neighbors
        #     #print(type(infer_param))
        #     parm = {}
        #     for name, parameters in infer_model.named_parameters():
        #         # print(name,':',parameters.size())
        #         parm[name]=parameters.detach().cpu().numpy()

        #     # s_ = parm['layers.1.linear.weight'].flatten()
        #     s = parm['layers.0.linear.weight'][0::]
        #     s = pca_svd(s,1).flatten()
        # else:
        #     # s[0] = acc_now
        #     # s = np.array(s)
        #     num_neighbors, fla= RL.choose_action(s)
        #     if fla == 1:
        #         num_neighbor = num_neighbors[0]+1
        #     else:
        #         num_neighbor = num_neighbors+1
        p1, time_cost1 = runGraph(model1,g1,args,optimizer1,labels1,train_nid1,cuda,args.num_neighbors)
        p2, time_cost2 = runGraph(model2,g2,args,optimizer2,labels2,train_nid2,cuda,args.num_neighbors)
        p3, time_cost3 = runGraph(model3,g3,args,optimizer3,labels3,train_nid3,cuda,args.num_neighbors)
        
        # time cost
        time_cost = round((time_cost1+time_cost2+time_cost3)/4,4)

        # aggregation
        for key, value in p2.items():  
            p1[key] = p1[key] * (len(train_nid1) / len(train_nid)) + p2[key] * (len(train_nid2) / len(train_nid)) + p3[key] * (len(train_nid3) / len(train_nid))
            # p1[key] = (p1[key] + p2[key] + p3[key])/3

        model1.load_state_dict(p1)
        model2.load_state_dict(p1)
        model3.load_state_dict(p1)

        for infer_param, param in zip(infer_model.parameters(), model1.parameters()):  
            infer_param.data.copy_(param.data)
            #print(type(infer_param))
        # parm = {}
        # for name, parameters in infer_model.named_parameters():
        #     #print(name,':',parameters.size())
        #     parm[name]=parameters.detach().cpu().numpy()

        # # s_ = parm['layers.1.linear.weight'].flatten()
        # s_ = parm['layers.0.linear.weight'][0::]
        # s_ = pca_svd(s_,1).flatten()
        #print(s_)
        
        # test 
        acc = inference(g_test,infer_model,args,labels_test,test_nid_test,in_feats,n_classes,n_test_samples_test,cuda)
        
        # # reward
        # # r = pow(1024,(acc-A))+math.log(max(time_cost_past-time_cost,0.001))
        # r = pow(1024,(acc-A))-1

        # # envs changed
        # RL.store_transition(s, num_neighbor-1, r, s_)
        # if (step > 20) and (step % 3 == 0):
        #     RL.learn()
        # time_cost_past = time_cost
        # step += 1
        # s = s_
        out.append(acc)
    time_end = time.time()
    print(round(time_end - time_now),4)
    dataframe = pd.DataFrame({'acc':out})
    dataframe.to_csv("acc_nondqn_10.csv",header = False,index=False,sep=',')
    # dataframe.to_csv("acc_dqn.csv",header = False,index=False,sep=',')
        

        



    


    # mutipule processes
    # task1 = mp.Process(target=runGraph,args=(model1,g1,train_mask,norm1,features1,args,optimizer1,labels1,train_nid1,pipe1[0],))
    # task2 = mp.Process(target=runGraph,args=(model2,g2,train_mask,norm2,features2,args,optimizer2,labels2,train_nid2,pipe2[0],))
    # task3 = mp.Process(target=runGraph,args=(model3,g3,train_mask,norm3,features3,args,optimizer3,labels3,train_nid3,pipe3[0],))
    # task4 = mp.Process(target=inference,args=(g_test,test_mask,norm_test,features_test,args,labels_test,test_nid_test,train_nid,train_nid1,train_nid2,train_nid3, in_feats,n_classes,n_test_samples,pipe1[1],pipe2[1],pipe3[1],))
    # #task4 = mp.Process(target=in        out.append(acc)ference,args=(g,args,labels,test_nid,train_nid,train_nid1,train_nid2,traing_nid3, in_feats,n_classes,n_test_samples,pipe1[1],pipe2[1],pipe3[1]))
    # #def                             inference(Graph,test_mask,norm,features,args,Labels,Test_nid,Train_nid,Train_nid1,Train_nid2,Train_nid3,In_feats,N_classes,N_test_samples,pipe_1,pipe_2,pipe_3)
    # processes = []

    # task1.start()

    # task2.start()

    # task3.start()

    # task4.start()

    # task1.join()
    # task2.join()
    # task3.join()
    # task4.join()


