import argparse, time, math
import gym
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
import torch.optim as optim
import sys
sys.path.append("/home/fahao/Py_code/ML-Distribution/Agent")
from DDPG import Agent
from sklearn.decomposition import PCA
from collections import deque
import matplotlib.pyplot as plt

acc=0

def pca_svd(data, k):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    # SVD
    U,S,V = torch.svd(torch.t(X))
    return torch.mm(X,U[:,:k])

# define the environment
class gcnEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):

        self.args = Gen_args(10)   # return the parameters
        
        # DQN parameter
        A = 0.6
        out=[0]
        acc_now = 0
        acc_next = 0 
        step = 0
        time_cost_past = 5

        # pubmed 
        node_list = list(range(19717))  
        list1 = node_list[0::3]
        list2 = node_list[1::3]
        list3 = node_list[2::3]
        list_test = node_list[0::1]

        
        # GCN parameter
        self.g, self.g1, self.g2, self.g3, self.g_test, self.norm1, self.norm2, self.norm3, self.norm_test, self.features1, self.features2, self.features3,\
            self.features_test, self.train_mask, self.test_mask, self.labels, self.labels1, self.labels2, self.labels3, self.labels_test, self.train_nid, \
                self.train_nid1, self.train_nid2, self.train_nid3, self.test_nid, self.test_nid_test, self.in_feats, self.n_classes, self.n_test_samples, \
                    self.n_test_samples_test = load_cora_data(list1, list2, list3, list_test,self.args)
        
        model1, self.optimizer1 = genGraph(self.args,self.in_feats,self.n_classes,self.n_test_samples,1)
        model2, self.optimizer2 = genGraph(self.args,self.in_feats,self.n_classes,self.n_test_samples,1)
        model3, self.optimizer3 = genGraph(self.args,self.in_feats,self.n_classes,self.n_test_samples,1)
        self.infer_model = genGraph(self.args,self.in_feats,self.n_classes,self.n_test_samples,2)
        self.local_model = [model1,model2,model3]

        if self.args.gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            self.local_model[0].cuda()
            self.local_model[1].cuda()
            self.local_model[2].cuda()
            self.infer_model.cuda()

            self.labels.cuda()
            self.labels1.cuda()
            self.labels2.cuda()
            self.labels3.cuda()
            self.labels_test.cuda()

        s = []
        s_ = []

        # Input Example

        



        self.action_space = 1 # 0, 1, 2，3，4: 不动，上下左右
        self.observation_space = 57


    def step(self, action):
        
        batch_sampling_method_1 = np.array([])
        batch_sampling_method_2 = np.array([])
        batch_sampling_method_3 = np.array([])
        test_batch_sampling_method = np.array([])
        layer_size = np.array([2,1])
        layer_scale = np.array([0.6,0.4])

        sampling_1 = action[0:2:1]
        sampling_2 = action[2:4:1]
        sampling_3 = action[4:6:1]
        #test sampling
        for layer in range(self.args.n_layers):
            for nodes in range(self.g_test.number_of_nodes()):
                test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)

        # training sampling
        # for layer in range(self.args.n_layers):
        #     for nodes in range(self.g1.number_of_nodes()):
        #         batch_sampling_method_1 = np.append(batch_sampling_method_1, layer_size[layer])
                # batch_sampling_method = np.append(batch_sampling_method, layer_size[layer])
                # batch_sampling_method = np.append(batch_sampling_method, layer_size[layer])
                


        
        # Scale training
        for layer in range(self.args.n_layers):
            for nodes in range(self.g.number_of_nodes()):
                temp1 = math.ceil(self.g.in_degree(nodes) * sampling_1[layer])
                batch_sampling_method_1 = np.append(batch_sampling_method_1, temp1)
            for nodes in range(self.g.number_of_nodes()):
                temp2 = math.ceil(self.g.in_degree(nodes) * sampling_2[layer])
                batch_sampling_method_2 = np.append(batch_sampling_method_1, temp2)
            for nodes in range(self.g.number_of_nodes()):
                temp3 = math.ceil(self.g.in_degree(nodes) * sampling_3[layer])
                batch_sampling_method_3 = np.append(batch_sampling_method_1, temp3)

        # for layer in range(args.n_layers):
        #     for nodes in range(g.number_of_nodes()):
        #         test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)
        
        
        time_now = time.time()
        p1, time_cost1, loss1 = runGraph(self.local_model[0],self.g1,self.args,self.optimizer1,self.labels1,self.train_nid1,self.cuda,batch_sampling_method_1)
        p2, time_cost2, loss2 = runGraph(self.local_model[1],self.g2,self.args,self.optimizer2,self.labels2,self.train_nid2,self.cuda,batch_sampling_method_2)
        p3, time_cost3, loss3 = runGraph(self.local_model[2],self.g3,self.args,self.optimizer3,self.labels3,self.train_nid3,self.cuda,batch_sampling_method_3)
        
        # get local model for state
        S_local = [0,0,0]
        for i in range(3):
            parm_local = {}
            for name, parameters in self.local_model[i].named_parameters():
                    #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
            s_1 = parm_local['layers.0.linear.weight'][0::]
            pca=PCA(n_components=2)
            pca.fit(s_1)  
            s_11 = pca.transform(s_1).flatten()
            s_2 = parm_local['layers.0.linear.bias'][0::]
            s_3 = parm_local['layers.1.linear.weight'][0::]
                # print(parm['layers.1.linear.weight'][0::])
            pca=PCA(n_components=2)
            pca.fit(s_3)  
            s_33 = pca.transform(s_3).flatten()
            s_4 = parm_local['layers.1.linear.bias'][0::]
            S_local[i] = np.concatenate((s_11,s_2,s_33,s_4),axis=0)

        # loss
        loss = (loss1 + loss2 + loss3)/3

        # time cost
        time_cost = round((time_cost1+time_cost2+time_cost3)/3,4)

        # aggregation
        for key, value in p1.items():  
            p1[key] = p1[key] * (len(self.train_nid1) / len(self.train_nid)) + p2[key] * (len(self.train_nid2) / len(self.train_nid)) + \
                p3[key] * (len(self.train_nid3) / len(self.train_nid))

        self.local_model[0].load_state_dict(p1)
        self.local_model[1].load_state_dict(p1)
        self.local_model[2].load_state_dict(p1)

        # test
        for infer_param, param in zip(self.infer_model.parameters(), self.local_model[0].parameters()):  
            infer_param.data.copy_(param.data)
        
        acc = inference(self.g_test,self.infer_model,self.args,self.labels_test,self.test_nid_test,self.in_feats,\
            self.n_classes,self.n_test_samples_test,self.cuda,test_batch_sampling_method)
        
        # get state
        # global model
        parm = {}
        for name, parameters in self.infer_model.named_parameters():
                #print(name,':',parameters.size())
            parm[name]=parameters.detach().cpu().numpy()
        s_1 = parm['layers.0.linear.weight'][0::]
        pca=PCA(n_components=2)
        pca.fit(s_1)  
        s_11 = pca.transform(s_1).flatten()
        s_2 = parm['layers.0.linear.bias'][0::]
        s_3 = parm['layers.1.linear.weight'][0::]
            # print(parm['layers.1.linear.weight'][0::])
        pca=PCA(n_components=2)
        pca.fit(s_3)  
        s_33 = pca.transform(s_3).flatten()
        s_4 = parm['layers.1.linear.bias'][0::]



        S_global = np.concatenate((s_11,s_2,s_33,s_4),axis=0)
    
        time_end = time.time()
        # print(loss,round(time_end-time_now,4))
        # print(time_cost)
        reward = pow(64,acc-0.8) - pow(256, 0 - 10*time_cost)
        # reward = pow(64,acc-0.8)
        
        S = np.concatenate((S_local[0],S_local[1],S_local[2],S_global),axis = 0)
        return S, reward, acc
    
    def reset(self):
        self.counts = 0
        S_local = [0,0,0]
        parm = {}

        for i in range(3):
            parm_local = {}
            for name, parameters in self.local_model[i].named_parameters():
                    #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
            s_1 = parm_local['layers.0.linear.weight'][0::]
            pca=PCA(n_components=2)
            pca.fit(s_1)  
            s_11 = pca.transform(s_1).flatten()
            s_2 = parm_local['layers.0.linear.bias'][0::]
            s_3 = parm_local['layers.1.linear.weight'][0::]
                # print(parm['layers.1.linear.weight'][0::])
            pca=PCA(n_components=2)
            pca.fit(s_3)  
            s_33 = pca.transform(s_3).flatten()
            s_4 = parm_local['layers.1.linear.bias'][0::]
            S_local[i] = np.concatenate((s_11,s_2,s_33,s_4),axis=0)

        for name, parameters in self.infer_model.named_parameters():
                #print(name,':',parameters.size())
            parm[name]=parameters.detach().cpu().numpy()
        s_1 = parm['layers.0.linear.weight'][0::]
        pca=PCA(n_components=2)
        pca.fit(s_1)  
        s_11 = pca.transform(s_1).flatten()
        s_2 = parm['layers.0.linear.bias'][0::]
        s_3 = parm['layers.1.linear.weight'][0::]
            # print(parm['layers.1.linear.weight'][0::])
        pca=PCA(n_components=2)
        pca.fit(s_3)  
        s_33 = pca.transform(s_3).flatten()
        s_4 = parm['layers.1.linear.bias'][0::]
        S_global = np.concatenate((s_11,s_2,s_33,s_4),axis=0)

        S = np.concatenate((S_local[0],S_local[1],S_local[2],S_global),axis = 0)
        return S
        
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None

#
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

#define the training process
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

# define the test process
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
def load_cora_data(list1, list2, list3, list_test, args):
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
    n_test_samples_test = n_test_samples

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

# train process
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,sampling):
    loss_fcn = nn.CrossEntropyLoss()

        # sampling
    time_now = time.time()
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            expand_factor = sampling,
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

# test process
def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda,sampling):

    num_acc = 0.

    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.test_batch_size,
                                                        expand_factor = sampling,
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
    # print('In round: ',epoch,' The Accuracy: ',acc)
    return acc

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime))
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

# generate the super-parameters
def Gen_args(num):
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.003,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000,
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
    out=[0]
    env = gcnEnv()
    agent = Agent(state_size=228, action_size=6, random_seed=2)
    print_every=100
    max_t = 10
    scores_deque = deque(maxlen=print_every)
    scores = []
    for episode in range(500):
        state = env.reset()
        agent.reset()
        score = 0
        
        action = agent.act(state)
        print('Take action: ',action)

        next_state, reward, acc = env.step(action)
        agent.step(state, action, reward, next_state)
        state = next_state
        score = reward
        print('Accuracy: ',acc)

        scores_deque.append(score)
        scores.append(score)
        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)), end="")
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,reward), end="\n")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        # if episode % print_every == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))



    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Total Return')
    plt.xlabel('Episode #')
    plt.show()
    

    dataframe = pd.DataFrame({'acc':out})
    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Reddit/acc_nondqn_10.csv",header = False,index=False,sep=',')

        

        



        # ...
        #  if acc >= 0.72:
        #      print('Training complete in round: ',epoch)
        #      break
        # ...

