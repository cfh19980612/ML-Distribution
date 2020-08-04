import argparse, time, math
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from functools import partial
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import citation_graph as citegrh
from dgl.data import RedditDataset
import pandas as pd
import networkx as nx
import threading
import psutil
import os
import time 
import datetime
import math
import sys
sys.path.append("/home/fahao/Py_code/ML-Distribution/Agent")
from DDPG import Agent
from sklearn.decomposition import PCA
from collections import deque
import matplotlib.pyplot as plt
import progressbar
import operator
from functools import reduce


# define the environment
class gcnEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
     
    def __init__(self):

        self.args = Gen_args(8)   # return the parameters

        # model and Opt list


        # DQN parameter
        A = 0.6
        out=[0]
        acc_now = 0
        acc_next = 0 
        step = 0
        time_cost_past = 5

        
        # GCN parameter
        self.g, self.train_mask,self.test_mask, \
            self.labels, self.train_nid, self.test_nid, \
                self.in_feats, self.n_classes, self.n_test_samples = load_cora_data(self.args)
        
        # initialize the model and Opt
        self.Model, self.Optimizer = genGraph(self.args,self.in_feats,self.n_classes,1)

        self.infer_model = genGraph(self.args,self.in_feats,self.n_classes,2)
        
        # gpu?
        if self.args.gpu < 0:
            self.cuda = False
        else:
            self.cuda = True


    def step(self, action, episode):
        
        Batch_sampling_method = []
        test_batch_sampling_method = np.array([])

        layer_size = np.array([2,1])
        Layer_scale = []
        Layer_scale = action[0::1]

        #test sampling
        for layer in range(self.args.n_layers + 1):
            for nodes in range(self.g.number_of_nodes()):
                test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)

        # Scale training
        for layer in range(self.args.n_layers + 1):
            for nodes in range(self.g.number_of_nodes()):
                temp = math.ceil(self.g.in_degree(nodes) * action[layer]*2)
                Batch_sampling_method = np.append(Batch_sampling_method, temp)

            


        P, Time_cost, Loss = runGraph(self.Model,self.g,self.args,self.Optimizer,\
            self.labels,self.train_nid,self.cuda,Batch_sampling_method)


#################################################################################################################################        
        # get local model for state
        parm_local = {}
        S_local = [None for i in range (2)]
        for i in range(2):
            S_local[i] = []
            for name, parameters in self.Model.named_parameters():
                        #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                    
            for a in parm_local['layers.0.linear.weight'][0::].flatten():
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.0.linear.bias'][0::]:
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.1.linear.weight'][0::].flatten():
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.1.linear.bias'][0::]:
                aa = a
                S_local[i].append(aa)
        pca=PCA(n_components=2)
        s_local=pca.fit_transform(S_local)

        # x = []
        # y = []
        # for j in range (1):
        #     x.append(s_local[j])
        #     y.append(s_local[j+1])
        # dataframe_dis = pd.DataFrame(x, columns=['X'])
        # dataframe_dis = pd.concat([dataframe_dis, pd.DataFrame(y,columns=['Y'])],axis=1)
        # dataframe_dis.to_csv("/home/fahao/Py_code/results/GCN-Pubmed(10)/distribution.csv",header = False,index=False,sep=',')
#################################################################################################################################


        # time cost
        time_cost = Time_cost


        # aggregation
        for infer_param, param in zip(self.infer_model.parameters(), self.Model.parameters()):  
            infer_param.data.copy_(param.data)
        # test
        acc = inference(self.g,self.infer_model,self.args,self.labels,self.test_nid,self.in_feats,\
            self.n_classes,self.n_test_samples,self.cuda,test_batch_sampling_method)
        
        # get state
#################################################################################################################################
        # global model
        parm = {}
        
        for name, parameters in self.infer_model.named_parameters():
            #print(name,':',parameters.size())
            parm[name]=parameters.detach().cpu().numpy()     
        S_global = [None for i in range (2)]
        for i in range (2):
            S_global[i] = []
            for a in parm['layers.0.linear.weight'][0::].flatten():
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.0.linear.bias'][0::]:
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.1.linear.weight'][0::].flatten():
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.1.linear.bias'][0::]:
                aa = a
                S_global[i].append(aa)
        pca=PCA(n_components=2)
        s_global=pca.fit_transform(S_global)
#################################################################################################################################



        reward = pow(32,acc-0.79) - math.log(1+15*time_cost)
        # reward = pow(10,acc-0.8) - 60*time_cost
        # reward = 0 - pow(64,0 - 100*time_cost)
        # reward = pow(64,acc-0.8)

        # next state
        S = s_global[0]
        S = np.concatenate((S,s_local[0]),axis=0)

        return S, reward, acc, time_cost
    




    def reset(self):
        parm = {}
        self.counts = 0
        parm_local = {}
        S_local = [None for i in range (2)]
        for i in range(2):
            S_local[i] = []
            for name, parameters in self.Model.named_parameters():
                        #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                    
            for a in parm_local['layers.0.linear.weight'][0::].flatten():
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.0.linear.bias'][0::]:
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.1.linear.weight'][0::].flatten():
                aa = a
                S_local[i].append(aa)
            for a in parm_local['layers.1.linear.bias'][0::]:
                aa = a
                S_local[i].append(aa)
        pca=PCA(n_components=2)
        s_local=pca.fit_transform(S_local)
        


        for name, parameters in self.infer_model.named_parameters():
                #print(name,':',parameters.size())
            parm[name]=parameters.detach().cpu().numpy()

        S_global = [None for i in range (2)]
        for i in range (2):
            S_global[i] = []
            for a in parm['layers.0.linear.weight'][0::].flatten():
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.0.linear.bias'][0::]:
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.1.linear.weight'][0::].flatten():
                aa = a
                S_global[i].append(aa)
            for a in parm['layers.1.linear.bias'][0::]:
                aa = a
                S_global[i].append(aa)
        pca=PCA(n_components=2)

        s_global=pca.fit_transform(S_global)

        S = s_global[0]
        S = np.concatenate((S,s_local[0]),axis=0)
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

# define the training process
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
def load_cora_data(args):
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
# train process
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
    # time_next = time.time()
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

# test process
def inference(Graph,infer_model,args,Labels,Test_nid,In_feats,N_classes,N_test_samples,cuda,sampling):

    num_acc = 0.
    if cuda:
        Labels.cuda()
        infer_model.cuda()
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

    if cuda:
        infer_model.cpu()
        Labels.cpu()
        torch.cuda.empty_cache()
    return acc

# training
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

# update target network
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

# generate the super-parameters
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


def dosomework():  
    time.sleep(0.01) 

if __name__ == '__main__':
    # target
    cora = 0.83
    citeseer = 0.72
    pubmed = 0.82
    reddit = 0.99 
    
    Y = []  # accuracy list
    X = []  # time cost list
    Z = []  # reward list
    times = 0
    env = gcnEnv()  # env
    agent = Agent(state_size=4, action_size=2, random_seed=2)  # agent
    print_every=100 
    max_t = 10
    scores_deque = deque(maxlen=print_every)
    scores = []
    max_value = 500
    for episode in range(1000):
        # initial env and agent
        if episode == 0:
            state = env.reset()
            agent.reset()
            score = 0

        # choose an action
        action = agent.act(state)
        
        # take an action 
        next_state, reward, acc, time_cost= env.step(action,episode)

        # store the experience and update the env
        agent.step(state, action, reward, next_state)
        state = next_state

        # store the total return
        score = reward
        scores_deque.append(score)
        scores.append(score)

        # print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode,reward), end="\n")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if episode > 0:
            times = times + time_cost
            X.append(times)
            Y.append(acc)
            Z.append(reward)

            # print {action || accuracy || timecost}
            print('---------------------------------------------------------------------------------------------------')
            print('\rEpisode {}\tAverage Score: {:.2f}\tAccuracy: {}\tTimecost: {:.4f}'.format(episode,reward,acc,times),flush=True)
            print('Take action:')
            print(action[0::1])
            # print(format(action[10::1],'^20'))
        if acc >= citeseer:
            break
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Total Return')
    plt.xlabel('Episode')
    plt.show()
    
    dataframe_reward = pd.DataFrame(X, columns=['X'])
    dataframe_reward = pd.concat([dataframe_reward, pd.DataFrame(Z,columns=['Z'])],axis=1)
    dataframes_reward = pd.DataFrame(Z, columns=['Z'])


    dataframes = pd.DataFrame(Y, columns=['Y'])
    dataframe = pd.DataFrame(X, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(Y,columns=['Y'])],axis=1)

    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/single/acc_gcn_ddpg.csv",header = False,index=False,sep=',')
    dataframes.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/single/acc_gcn_ddpg(round).csv",header = False,index=False,sep=',')
    dataframe_reward.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/single/reward.csv",header = False,index=False,sep=',')
    dataframes_reward.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/single/reward(round).csv",header = False,index=False,sep=',')





        
