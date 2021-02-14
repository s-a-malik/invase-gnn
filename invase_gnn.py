"""Instance-wise Variable Selection module with Graph Neural Networks

Pytorch Implementation of INVASE (https://openreview.net/forum?id=BJg_roAcK7)
with extension to work with GNNs.
"""

# import packages
from tqdm import trange

import numpy as np

import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import subgraph

from utils import AverageMeter, prediction_performance_metric, feature_performance_metric

class InvaseGNN(nn.Module):
    """Main INVASE-GNN model class

    Parameters:
    - actor_h_dim: hidden state dimensions for actor
    - critic_h_dim: hidden state dimensions for critic
    - n_layer: the number of graph conv layers
    - activation: activation function of models
    """
    def __init__(self, fea_dim, label_dim, actor_h_dim, critic_h_dim, n_layer, node_lamda, fea_lamda, dropout):
        super(InvaseGNN, self).__init__()
        
        self.actor_h_dim = actor_h_dim
        self.critic_h_dim = critic_h_dim
        self.n_layer = n_layer
        self.fea_dim = fea_dim 
        self.label_dim = label_dim
        self.lamda1 = node_lamda
        self.lamda2 = fea_lamda
        self.dropout = dropout

        self.actor = Actor(self.fea_dim, self.n_layer, self.actor_h_dim, self.dropout)
        self.critic = Critic(self.fea_dim, self.n_layer, self.critic_h_dim, self.label_dim, self.dropout)
        self.baseline = Baseline(self.fea_dim, self.n_layer, self.critic_h_dim, self.label_dim, self.dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch, component="actor"):
        if component == "actor":
            return self.actor(x, edge_index, batch)
        elif component == "critic":
            return self.critic(x, edge_index, batch)
        elif component == "baseline":
            return self.baseline(x, edge_index, batch)
        else:
            raise NotImplementedError("This was not supposed to be used.")  

    def predict(self, data):
        """End to end prediction for INVASE with input batched graph
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Generate a batch of selection probability

        # pass through selector
        node_prob, fea_prob = self(x, edge_index, batch, component="actor")
        # Sampling the features based on the selection_probability
        node_selection_mask = torch.bernoulli(node_prob)       
        node_selection = torch.squeeze(torch.nonzero(node_selection_mask, as_tuple=False))
        fea_selection_mask = torch.bernoulli(fea_prob)
        # make subgraph
        # mask out features
        subgraph_x = x * fea_selection_mask[batch]  # keep all the nodes
        subgraph_edge_index, _ = subgraph(node_selection, edge_index)  # returning only the edges of the subgraph
        # Prediction
        y_hat = self.critic([subgraph_x, node_selection], subgraph_edge_index, batch)
        # unnormalised - need to softmax
        return y_hat

    def actor_loss(self, node_selection_mask, fea_selection_mask, batch_idx, critic_out, baseline_out, y_true, node_pred, fea_pred):

        y_true_one_hot = nn.functional.one_hot(y_true.long(), num_classes=self.label_dim)
        
        critic_loss = -torch.sum(y_true_one_hot * torch.log(critic_out + 1e-8), dim=1)
        baseline_loss = -torch.sum(y_true_one_hot * torch.log(baseline_out + 1e-8), dim=1)
        reward = -(critic_loss - baseline_loss)

        # Policy gradient loss computation.
        # for nodes, get graphwise loss - this depends on size of graphs in batch
        custom_actor_loss = reward * scatter(node_selection_mask * torch.log(node_pred + 1e-8) + (1 - node_selection_mask) * torch.log(1 - node_pred + 1e-8), 
                                                batch_idx, reduce="sum")

        custom_actor_loss -= self.lamda1 * scatter(node_pred, batch_idx, reduce="mean") #normalise by number of graphs in batch
        
        # add loss for features
        custom_actor_loss += \
            reward * torch.sum(fea_selection_mask * torch.log(fea_pred + 1e-8) + (1 - fea_selection_mask) * torch.log(1 - fea_pred + 1e-8), dim=1)

        custom_actor_loss -= self.lamda2 * torch.mean(fea_pred, dim=1)  #l0 loss normalised?
        custom_actor_loss = torch.mean(-custom_actor_loss)  # mean over batch

        return custom_actor_loss

    def importance_score(self, data):
        """Return node and feature importance score.
        
        Args:
        - data: graph batch
        
        Returns:
        - node_importance: instance-wise node importance for nodes in batch
        - feature_importance: instance-wise (per graph) feature importance.
        """        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_importance, feature_importance = self.actor(x, edge_index, batch) 

        return np.asarray(feature_importance), np.asarray(node_importance)

    def evaluate(self, generator, criterion, optimizer, device, task="train"):
        """evaluate the model
        Params:
        generator - dataloader
        criterion - baseline loss function 
        optimiser - optimiser linked to model parameters
        device - cuda or cpu
        task - train, val or test
        """
        actor_loss_meter = AverageMeter()
        baseline_acc_meter = AverageMeter()
        critic_acc_meter = AverageMeter()
        prop_of_nodes = AverageMeter()
        prop_of_feas = AverageMeter()

        if task == "test":
            self.eval()
            x_test = []
            selected_features = []
            selected_nodes = []
            y_trues = []
            y_preds = []
        else:
            if task == "val":
                self.eval()
            elif task == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as task")

        with trange(len(generator)) as t:
            for data in generator:
                # these are batched graphs
                orig = data.clone()
                x, edge_index, batch, y_true = data.x, data.edge_index, data.batch, data.y
                x, edge_index, batch, y_true = x.to(device), edge_index.to(device), batch.to(device), y_true.to(device)
                # prediction on full graph
                baseline_logits = self(x, edge_index, batch, component="baseline")
                # print(baseline_logits)
                baseline_loss = criterion(baseline_logits, y_true)  

                # pass through selector
                node_prob, fea_prob = self(x, edge_index, batch, component="actor")
                # Sampling the features based on the selection_probability
                node_selection_mask = torch.bernoulli(node_prob)       
                node_selection = torch.squeeze(torch.nonzero(node_selection_mask, as_tuple=False))
                fea_selection_mask = torch.bernoulli(fea_prob)

                # make subgraph
                # mask out features
                subgraph_x = x * fea_selection_mask[batch]  # keep all the nodes
                subgraph_edge_index, _ = subgraph(node_selection, edge_index)  # returning only the edges of the subgraph

                critic_logits = self([subgraph_x, node_selection], subgraph_edge_index, batch, component="critic")
                critic_loss = criterion(critic_logits, y_true)  
                
                actor_loss = self.actor_loss(node_selection_mask.clone().detach(), 
                                            fea_selection_mask.clone().detach(),
                                            batch.clone().detach(),
                                            self.softmax(critic_logits).clone().detach(), 
                                            self.softmax(baseline_logits).clone().detach(), 
                                            y_true.float(), 
                                            node_prob, 
                                            fea_prob)

                actor_loss_meter.update(actor_loss.data.cpu().item(), y_true.size(0))
                critic_preds = torch.argmax(critic_logits, dim=1)
                critic_acc = torch.sum(critic_preds==y_true).float() / y_true.size(0)
                critic_acc_meter.update(critic_acc)
                baseline_preds = torch.argmax(baseline_logits, dim=1)
                baseline_acc = torch.sum(baseline_preds==y_true).float() / y_true.size(0)
                baseline_acc_meter.update(baseline_acc)
                
                prop_of_feas.update(torch.mean(torch.mean(fea_selection_mask, dim=-1)), y_true.size(0))
                prop_of_nodes.update(torch.mean(node_selection_mask), y_true.size(0))

                if task == "test":
                    # collect and analyse results
                    x_test += orig.to_data_list()
                    selected_features.append(fea_prob.detach().cpu().numpy())
                    
                    # need to change to batchwise
                    node_prob = node_prob.detach().cpu().numpy()
                    selected_nodes += [[x for j, x in enumerate(node_prob) if batch[j] == i] for i in range(len(y_true))]
                    y_trues.append(y_true.detach().cpu().numpy())
                    y_preds.append(critic_preds.detach().cpu().numpy())

                else:

                    if task == "train":
                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        total_loss = actor_loss + critic_loss + baseline_loss
                        total_loss.backward()
                        optimizer.step()
                t.update()
        
        if task == "test":
            # auc, apr, acc = prediction_performance_metric(y_pred, y_true)
            # # Print the performance of feature importance    
            # print('AUC: ' + str(np.round(auc, 3)) + \
            #         ', APR: ' + str(np.round(apr, 3)) + \
            #         ', ACC: ' + str(np.round(acc, 3)))
            return critic_acc_meter.avg, baseline_acc_meter.avg, x_test, \
                    np.concatenate(selected_features, axis=0), selected_nodes, np.concatenate(y_trues), np.concatenate(y_preds)
        else:
            return actor_loss_meter.avg, critic_acc_meter.avg, baseline_acc_meter.avg, prop_of_feas.avg, prop_of_nodes.avg


class Actor(nn.Module):
    """Selector network. Output probabilities of 
    selecting each node AND features.
    """
    def __init__(self, fea_dim, n_layer, actor_h_dim, dropout):
        super(Actor, self).__init__()
        self.convs = nn.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.fea_lin1 = nn.Linear(fea_dim, actor_h_dim)
        self.fea_lin2 = nn.Linear(actor_h_dim, fea_dim)
        self.node_lin = nn.Linear(fea_dim, 1)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Apply a final classifier
        node_prob = torch.squeeze(self.node_lin(x))            # [batch_num_nodes]

        # 4. get feature importance for graph
        fea_prob = global_mean_pool(x, batch)                  
        fea_prob = self.act(self.fea_lin1(fea_prob))           # [batch_size, actor_h_dim]
        fea_prob = self.fea_lin2(fea_prob)                     # [batch_size, fea_dim]

        return self.sigmoid(node_prob), self.sigmoid(fea_prob)

class Critic(nn.Module):
    def __init__(self, fea_dim, n_layer, critic_h_dim, label_dim, dropout):
        super(Critic, self).__init__()
        self.convs = nn.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.lin1 = nn.Linear(fea_dim, critic_h_dim)
        self.lin2 = nn.Linear(critic_h_dim, label_dim)
        self.act = nn.ReLU()
        self.dropout = dropout

    def forward(self, x_comb, edge_index, batch):
        
        x, node_selection = x_comb
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Readout layer
        # remove masked nodes
        x_selected = x[node_selection]
        batch_selected = batch[node_selection]
        # out = torch.zeros((, x.shape[-1]))
        out = scatter(x_selected, batch_selected, dim=0, reduce="mean", dim_size=batch[-1]+1) # [batch_size, fea_dim]
        # x = global_mean_pool(x, batch)       

        # 3. Apply a final classifier
        out = self.act(self.lin1(out))           # [batch_size, critic_h_dim]
        out = self.lin2(out)                     # [batch_size, label_dim]
        return out

class Baseline(nn.Module):
    def __init__(self, fea_dim, n_layer, critic_h_dim, label_dim, dropout):
        super(Baseline, self).__init__()
        self.convs = nn.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.lin1 = nn.Linear(fea_dim, critic_h_dim)
        self.lin2 = nn.Linear(critic_h_dim, label_dim)
        self.act = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, fea_dim]

        # 3. Apply a final classifier
        x = self.act(self.lin1(x))           # [batch_size, critic_h_dim]
        x = self.lin2(x)                # [batch_size, label_dim]
        return x
