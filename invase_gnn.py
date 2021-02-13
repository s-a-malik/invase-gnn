"""Instance-wise Variable Selection module with Graph Neural Networks

Pytorch Implementation of INVASE (https://openreview.net/forum?id=BJg_roAcK7)
with extension to work with GNNs.
"""

# import packages
import numpy as np

import torch
import torch.nn as nn

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
    def __init__(self, fea_dim, label_dim, actor_h_dim, critic_h_dim, n_layer):

        super(InvaseGNN, self).__init__()
        
        self.actor_h_dim = actor_h_dim
        self.critic_h_dim = critic_h_dim
        self.n_layer = n_layer
        self.fea_dim = fea_dim 
        self.label_dim = label_dim
        # Build the selector , n_layer, actor_h_dim, num_nodes
        self.actor = Actor(self.fea_dim, self.n_layer, self.actor_h_dim)
        # Build the predictor
        self.critic = Critic(self.fea_dim, self.n_layer, self.critic_h_dim, self.label_dim)
        self.baseline = Baseline(self.fea_dim, self.n_layer, self.critic_h_dim, self.label_dim)
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
        node_prob, fea_prob = self.actor(x, edge_index, batch)
        # Sampling the features based on the selection_probability
        print(node_prob)
        node_selection = torch.bernoulli(node_prob)
        print(node_selection)         
        node_selection = torch.nonzero(node_selection)
        print(node_selection)
        fea_selection = torch.bernoulli(fea_prob)
        print(fea_selection)
        # make subgraph
        subgraph_edge_index = subgraph(node_selection, data.edge_index)
        print(subgraph_edges)
        # mask out features
        subgraph_x = x[node_selection] * fea_selection
        print(subgraph_x)
        subgraph_batch = batch[node_selection]
        print(subgraph_batch)
        # Prediction
        y_hat = self.critic(subgraph_x, subgraph_edge_index, subgraph_batch)
        print(y_hat)

        # unnormalised - need to softmax
        return y_hat

    def actor_loss(self, node_selection_mask, fea_selection_mask, critic_out, baseline_out, y_true, node_pred, fea_pred):
        
        y_true_one_hot = nn.functional.one_hot(y_true, num_classes=self.label_dim)
        print(y_true_one_hot.shape)
        print(critic_out.shape)
        print(baseline_out.shape)
        critic_loss = -torch.sum(y_true_one_hot * torch.log(critic_out + 1e-8), dim=1)
        baseline_loss = -torch.sum(y_true_one_hot * torch.log(baseline_out + 1e-8), dim=1)
        reward = -(critic_loss - baseline_loss)
        # Policy gradient loss computation.
        # for nodes
        custom_actor_loss = \
            reward * torch.sum(node_selection_mask * torch.log(node_pred + 1e-8) + (1 - node_selection_mask) * torch.log(1 - node_pred + 1e-8), dim=1)

        custom_actor_loss -= self.lamda * torch.mean(node_pred, dim=1)  #l0 loss normalised?
        
        # for features
        custom_actor_loss += \
            reward * torch.sum(fea_selection_mask * torch.log(fea_pred + 1e-8) + (1 - fea_selection_mask) * torch.log(1 - fea_pred + 1e-8), dim=1)

        custom_actor_loss -= self.lamda * torch.mean(fea_pred, dim=1)  #l0 loss normalised?

        custom_actor_loss = torch.mean(-custom_actor_loss)

        return custom_actor_loss

    def importance_score(self, data):
        """Return node and feature importance score.
        
        Args:
        - data: graph batch
        
        Returns:
        - node_importance: instance-wise node importance for nodes in batch
        - feature_importance: instance-wise (per graph) feature importance.
        """        
        node_importance, feature_importance = self.actor(data) 

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

        if task == "test":
            self.eval()
            x_test = []
            selected_features = []
            selected_nodes = []
            y_trues = []
            y_preds = []
            critic_test_acc_meter = AverageMeter()
            baseline_test_acc_meter = AverageMeter()
        else:
            actor_loss_meter = AverageMeter()
            baseline_acc_meter = AverageMeter()
            critic_acc_meter = AverageMeter()
            if task == "val":
                self.eval()
            elif task == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as task")

        with trange(len(generator)) as t:
            for data in generator:
                # these are batched graphs
                data = data.to(device)
                x, edge_index, batch, y_true = data.x, data.edge_index, data.batch, data.y

                # prediction on full graph
                baseline_logits = self(x, edge_index, batch, component="baseline")
                print(baseline_logits.shape)
                print(y_true.shape)
                baseline_loss = criterion(baseline_logits, y_true)  

                # pass through selector
                node_prob, fea_prob = self(x, edge_index, batch, component="actor")

                # Sampling the features based on the selection_probability
                print(node_prob)
                node_selection_mask = torch.bernoulli(node_prob)
                print(node_selection)         
                node_selection = torch.nonzero(node_selection_mask)
                print(node_selection)
                fea_selection_mask = torch.bernoulli(fea_prob)
                print(fea_selection_mask)
                # make subgraph
                subgraph_edge_index = subgraph(node_selection, data.edge_index)
                print(subgraph_edges)
                # mask out features
                subgraph_x = x[node_selection] * fea_selection_mask
                print(subgraph_x)
                subgraph_batch = batch[node_selection]
                print(subgraph_batch)

                # prediction on selected subgraph
                critic_logits = self(subgraph_x, subgraph_edge_index, subgraph_batch, component="critic")
                print(critic_logits)
                critic_loss = criterion(critic_logits, y_true)  
                
                actor_loss = self.actor_loss(node_selection_mask.clone().detach(), 
                                            fea_selection_mask.clone().detach(),
                                            self.softmax(critic_logits).clone().detach(), 
                                            self.softmax(baseline_logits).clone().detach(), 
                                            y_true.float(), 
                                            node_prob, 
                                            fea_prob)

                actor_loss_meter.update(actor_loss.data.cpu().item(), y_true.size(0))
                critic_preds = torch.argmax(critic_logits, dim=1)
                critic_acc = torch.sum(critic_preds==y_true)/len(y_true)
                critic_acc_meter.update(acc)
                baseline_preds = torch.argmax(baseline_logits, dim=1)
                baseline_acc = torch.sum(baseline_preds==y_true)/len(y_true)
                baseline_acc_meter.update(acc)

                if task == "test":
                    # collect and analyse results
                    critic_test_acc_meter.update(acc)
                    baseline_test_acc_meter.update(acc)
                    x_test.append(data)
                    selected_features.append(fea_prob)
                    selected_nodes.append(node_prob)
                    y_trues.append(y_true)
                    y_preds.append(critic_preds)

                else:

                    # preds = torch.argmax(logits, dim=1)
                    # acc = torch.sum(preds==y_batch)/len(y_batch)
                    # acc_meter.update(acc)
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
            return critic_test_acc_meter.avg, baseline_test_acc_meter.avg, x_test, \
                    selected_features, selected_nodes, y_trues, y_preds
        else:
            return actor_loss_meter.avg, critic_acc_meter.avg, baseline_acc_meter.avg


class Actor(nn.Module):
    """Selector network. Output logits corresponding to probabilities of 
    selecting each node AND features.
    """
    def __init__(self, fea_dim, n_layer, actor_h_dim):
        super(Actor, self).__init__()
        self.convs = n.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.fea_lin1 = nn.Linear(fea_dim, actor_h_dim)
        self.fea_lin2 = nn.Linear(actor_h_dim, fea_dim)
        self.node_lin = nn.Linear(fea_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = self.act(x)

        # 3. Apply a final classifier
        node_prob = torch.squeeze(self.node_lin(x))            # [batch_num_nodes]
        print(node_prob.shape)

        # 4. get feature importance for graph
        fea_prob = global_mean_pool(x, batch)                  
        fea_prob = self.act(self.fea_lin1(fea_prob))           # [batch_size, actor_h_dim]
        fea_prob = self.fea_lin2(fea_prob)                     # [batch_size, fea_dim]

        return node_prob, fea_prob

class Critic(nn.Module):
    def __init__(self, fea_dim, n_layer, critic_h_dim, label_dim):
        super(Critic, self).__init__()
        self.convs = n.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.lin1 = nn.Linear(fea_dim, critic_h_dim)
        self.lin2 = nn.Linear(critic_h_dim, label_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = act(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)       # [batch_size, fea_dim]

        # 3. Apply a final classifier
        x = self.act(self.lin1(x))           # [batch_size, critic_h_dim]
        x = self.lin2(x)                     # [batch_size, label_dim]
        return x

class Baseline(nn.Module):
    def __init__(self, fea_dim, n_layer, critic_h_dim, label_dim):
        super(Baseline, self).__init__()
        self.convs = n.ModuleList(
                        [GCNConv(fea_dim, fea_dim) for i in range(n_layer)])
        self.lin1 = nn.Linear(fea_dim, critic_h_dim)
        self.lin2 = nn.Linear(critic_h_dim, label_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # graph convolutions
        for graph_func in self.convs:
            x = graph_func(x, edge_index)
            x = act(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, fea_dim]

        # 3. Apply a final classifier
        x = self.act(self.lin1(x))           # [batch_size, critic_h_dim]
        x = self.lin2(x)                # [batch_size, label_dim]
        return x
