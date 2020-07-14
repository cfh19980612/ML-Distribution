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
import time
import pandas as pd

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
# initialize model and optimizer
def genModel(args,In_feats,N_classes,flag):
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

# training 
def training(args, g, model, optimizer, train_nid, labels, cuda):
    # initialize the loss function
    loss_fcn = nn.CrossEntropyLoss()

    # initialize graph
    time_now = time.time()
    for nf in dgl.contrib.sampling.NeighborSampler(g, args.batch_size,
                                                       args.num_neighbors,
                                                       neighbor_type='in',
                                                       shuffle=True,
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=train_nid):
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
    time_end = time.time()
    times = round(time_end-time_now,4)
    p = model.state_dict()
    return p, times

# inference
def inference(args, g, model, test_nid, labels, n_test_samples):

        # for infer_param, param in zip(infer_model.parameters(), model.parameters()):
        #     infer_param.data.copy_(param.data)

    num_acc = 0.

    for nf in dgl.contrib.sampling.NeighborSampler(g, args.test_batch_size,
                                                       g.number_of_nodes(),
                                                       neighbor_type='in',
                                                       num_workers=32,
                                                       num_hops=args.n_layers+1,
                                                       seed_nodes=test_nid):
        nf.copy_from_parent()
        model.eval()
        with torch.no_grad():
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(device=pred.device, dtype=torch.long)
            batch_labels = labels[batch_nids]
            num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

    print("Test Accuracy {:.4f}". format(num_acc/n_test_samples))
    acc = round((num_acc/n_test_samples),4)
    return acc


def Gen_args():
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
            help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=300,
            help="batch size")
    parser.add_argument("--test-batch-size", type=int, default=5000,
            help="test batch size")
    parser.add_argument("--num-neighbors", type=int, default=2,
            help="number of neighbors to be sampled")
    parser.add_argument("--n-hidden", type=int, default=32,
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
    # initialize the Hyperparameter
    args = Gen_args()
    # load and preprocess dataset
    data = citegrh.load_citeseer()

    # initialize the time (X)
    times = 0
    X = []
    # initialize the accuracy (Y)
    Y = []

    node_list = list(range(3327))  
    list1 = node_list[0::3]
    list2 = node_list[1::3]
    list3 = node_list[2::3]

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

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              n_train_samples,
              n_val_samples,
              n_test_samples))

    # create GCN model
    g = DGLGraph(data.graph, readonly=True)
    norm = 1. / g.in_degrees().float().unsqueeze(1)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        norm = norm.cuda()

    g.ndata['features'] = features

    num_neighbors = args.num_neighbors

    g.ndata['norm'] = norm

    model1, optimizer1 = genModel(args,in_feats,n_classes,1)
    model2, optimizer2 = genModel(args,in_feats,n_classes,1)
    model3, optimizer3 = genModel(args,in_feats,n_classes,1)

    infer_model = genModel(args,in_feats,n_classes,2)


    if cuda:
        model1.cuda()
        model2.cuda()
        model3.cuda()
        infer_model.cuda()



    train_nid1 = []
    train_nid2 = []
    train_nid3 = []
    for i in range(len(list1)):
        if list1[i] in train_nid:
            train_nid1.append(list1[i])
    train_nid1 = np.array(train_nid1)

    for i in range(len(list2)):
        if list2[i] in train_nid:
            train_nid2.append(list2[i])
    train_nid2 = np.array(train_nid2)

    for i in range(len(list3)):
        if list3[i] in train_nid:
            train_nid3.append(list3[i])
    train_nid3 = np.array(train_nid3)


    for epoch in range(args.n_epochs):
        p1, time1 = training(args, g, model1, optimizer1, train_nid1, labels, cuda)
        p2, time2 = training(args, g, model2, optimizer2, train_nid2, labels, cuda)
        p3, time3 = training(args, g, model3, optimizer3, train_nid3, labels, cuda)

        time_cost = round((time1 + time2 + time3)/3,4)
        # aggregation
        for key, value in p1.items():  
            p1[key] = p1[key] * (len(train_nid1) / len(train_nid)) + p2[key] * (len(train_nid2) / len(train_nid)) + \
                p3[key] * (len(train_nid3) / len(train_nid))

        model1.load_state_dict(p1)
        model2.load_state_dict(p1)
        model3.load_state_dict(p1)

        # test
        for infer_param, param in zip(infer_model.parameters(), model1.parameters()):  
            infer_param.data.copy_(param.data)
        
        acc = inference(args, g, infer_model, test_nid, labels, n_test_samples)

        if epoch > 0:
            times = times + time_cost
            X.append(times)
            Y.append(acc)
            print('Epoch: ',epoch,'||', 'Accuracy: ', acc, '||', 'Timecost: ', times)
        
    dataframe = pd.DataFrame(X, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(Y,columns=['Y'])],axis=1)

    dataframe.to_csv("/home/fahao/Py_code/results/GCN-Citeseer/acc_gcn_GraphSAGE.csv",header = False,index=False,sep=',')



    
    

