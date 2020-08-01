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

        self.args = Gen_args(8)   # return the parameters
        self.num_clients = 8

        # model and Opt list
        self.Model = [None for i in range (self.num_clients)]
        self.Optimizer = [None for i in range (self.num_clients)]

        # DQN parameter
        A = 0.6
        out=[0]
        acc_now = 0
        acc_next = 0 
        step = 0
        time_cost_past = 5

        # Client graph list and test
        node_list = list(range(3327))
        Client = [None for i in range (self.num_clients)]
        for i in range(self.num_clients):
            Client[i] = node_list[i::self.num_clients]
        list_test = node_list[0::1]

        
        # GCN parameter
        self.g, self.g_test,self.norm_test,self.features_test,self.train_mask,self.test_mask, \
            self.labels, self.labels_test, self.train_nid, self.Train_nid, self.test_nid, self.test_nid_test, \
                self.in_feats, self.n_classes, self.n_test_samples, self.n_test_samples_test = load_cora_data(self.args, Client, list_test, self.num_clients)
        
        # initialize the model and Opt
        for i in range(self.num_clients):
            self.Model[i], self.Optimizer[i] = genGraph(self.args,self.in_feats,self.n_classes,1)

        self.infer_model = genGraph(self.args,self.in_feats,self.n_classes,2)
        
        # gpu?
        if self.args.gpu < 0:
            self.cuda = False
        else:
            self.cuda = True
            for i in range(self.num_clients):
                self.Model[i].cuda()
            self.infer_model.cuda()

            self.labels.cuda()
            self.labels_test.cuda()

        s = []
        s_ = []

        # Input Example

        



        self.action_space = 6 
        self.observation_space = 468


    def step(self, action, episode):
        
        s_1 = np.array([])
        s_2 = np.array([])
        s_3 = np.array([])
        s_4 = np.array([])
        s_5 = np.array([])
        s_6 = np.array([])
        s_7 = np.array([])
        s_8 = np.array([])
        s_9 = np.array([])
        s_0 = np.array([])
        Batch_sampling_method = [s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9]
        test_batch_sampling_method = np.array([])

        layer_size = np.array([2,1])
        Layer_scale = [None for i in range (self.num_clients)]
        j = 0
        for i in range(self.num_clients):
            Layer_scale[i] = action[j:j+2:1]
            j = i+2

        #test sampling
        for layer in range(self.args.n_layers + 1):
            for nodes in range(self.g_test.number_of_nodes()):
                test_batch_sampling_method = np.append(test_batch_sampling_method, 10000)

        # Scale training
        for i in range(self.num_clients):
            for layer in range(self.args.n_layers + 1):
                for nodes in range(self.g.number_of_nodes()):
                    temp = math.ceil(self.g.in_degree(nodes) * Layer_scale[i][layer])
                    Batch_sampling_method[i] = np.append(Batch_sampling_method[i], temp)

        P = [None for i in range (self.num_clients)]
        Time_cost = [None for i in range (self.num_clients)]
        Loss = [None for i in range (self.num_clients)]
        for i in range(self.num_clients):
            P[i], Time_cost[i], Loss[i] = runGraph(self.Model[i],self.g,self.args,self.Optimizer[i],\
                self.labels,self.Train_nid[i],self.cuda,Batch_sampling_method[i])


#################################################################################################################################        
        # get local model for state
        S_local = [None for i in range(self.num_clients)]
        for i in range(self.num_clients):
            parm_local = {}
            for name, parameters in self.Model[i].named_parameters():
                    #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                
            S_local[i] = []
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

        x = []
        y = []
        for i in range (self.num_clients):
            for j in range (1):
                x.append(s_local[i,j])
                y.append(s_local[i,j+1])
        dataframe_dis = pd.DataFrame(x, columns=['X'])
        dataframe_dis = pd.concat([dataframe_dis, pd.DataFrame(y,columns=['Y'])],axis=1)
        dataframe_dis.to_csv("/home/fahao/Py_code/results/GCN-Pubmed(10)/distribution.csv",header = False,index=False,sep=',')
#################################################################################################################################


        # time cost
        time_cost = 0
        for i in range(self.num_clients):
            time_cost += Time_cost[i]
        time_cost = round(time_cost/self.num_clients,4)

        # aggregation
        for key, value in P[0].items():  
            for i in range(self.num_clients):
                if i == 0:
                    P[0][key] = P[i][key] * (len(self.Train_nid[i]) / len(self.train_nid))
                else:
                    P[0][key] += P[i][key] * (len(self.Train_nid[i]) / len(self.train_nid))
        
        for i in range(self.num_clients):
            self.Model[i].load_state_dict(P[0])

        # aggregation
        for infer_param, param in zip(self.infer_model.parameters(), self.Model[0].parameters()):  
            infer_param.data.copy_(param.data)
        # test
        acc = inference(self.g_test,self.infer_model,self.args,self.labels_test,self.test_nid_test,self.in_feats,\
            self.n_classes,self.n_test_samples_test,self.cuda,test_batch_sampling_method)
        
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



        reward = pow(32,acc-0.7) - math.log(1 + 30*time_cost)
        # reward = pow(10,acc-0.8) - 60*time_cost
        # reward = 0 - pow(64,0 - 100*time_cost)
        # reward = pow(64,acc-0.8)
        
        # next state
        S = s_global[0]
        for i in range(self.num_clients):
            S = np.concatenate((S,s_local[i]),axis=0)

        return S, reward, acc, time_cost
    




    def reset(self):
        self.counts = 0
        S_local = [None for i in range (self.num_clients)]
        parm = {}
        for name, parameters in self.Model[0].named_parameters():
            print(name,':',parameters.size())
        s = [None for i in range(self.num_clients)]
        for i in range(self.num_clients):
            parm_local = {}
            for name, parameters in self.Model[i].named_parameters():
                    #print(name,':',parameters.size())
                parm_local[name]=parameters.detach().cpu().numpy()
                
            s[i] = []
            for a in parm_local['layers.0.linear.weight'][0::].flatten():
                aa = a
                s[i].append(aa)
            for a in parm_local['layers.0.linear.bias'][0::]:
                aa = a
                s[i].append(aa)
            for a in parm_local['layers.1.linear.weight'][0::].flatten():
                aa = a
                s[i].append(aa)
            for a in parm_local['layers.1.linear.bias'][0::]:
                aa = a
                s[i].append(aa)           
        pca=PCA(n_components=2)
        S_local_new=pca.fit_transform(s)
        


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
        for i in range(self.num_clients):
            S = np.concatenate((S,S_local_new[i]),axis=0)
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
def load_cora_data(args, Client, list_test, num_clients):
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

# train process
def runGraph(Model,Graph,args,Optimizer,Labels,train_nid,cuda,sampling):
    loss_fcn = nn.CrossEntropyLoss()

        # sampling
    # time_now = time.time()
    time_cost = 0
    for nf in dgl.contrib.sampling.NeighborSampler(Graph, args.batch_size,  
                                                            expand_factor = sampling,
                                                            neighbor_type='in',
                                                            shuffle=True,
                                                            num_workers=32,
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


def dosomework():  
    time.sleep(0.01) 

if __name__ == '__main__':
    # target
    cora = 0.83
    citeseer = 0.72
    pubmed = 0.82

    Y = []  # accuracy list
    X = []  # time cost list
    Z = []  # reward list
    times = 0
    env = gcnEnv()  # env
    agent = Agent(state_size=18, action_size=20, random_seed=2)  # agent
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
            print(action[0:10:1])
            print(action[10::1])
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

    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/acc_gcn_ddpg.csv",header = False,index=False,sep=',')
    dataframes.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/acc_gcn_ddpg(round).csv",header = False,index=False,sep=',')
    dataframe_reward.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/reward.csv",header = False,index=False,sep=',')
    dataframes_reward.to_csv("/home/fahao/Py_code/results/GCN-Citeseer(8)/reward(round).csv",header = False,index=False,sep=',')





        
