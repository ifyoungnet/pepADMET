import datetime
from sklearn.metrics import roc_auc_score, mean_squared_error,mean_absolute_error, precision_recall_curve, auc, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import pandas as pd
import ast
from utils import weight_visualization


class WeightAndSum(nn.Module):   
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention  
        self.in_feats = in_feats    
        self.task_num = task_num       
        self.return_weight=return_weight    
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])  
        self.shared_weighting = self.atom_weight(self.in_feats)  
    def forward(self, bg, feats):  
        feat_list = []
        atom_list = []
        # cal specific feats  
        for i in range(self.task_num):
            with bg.local_scope():   
                bg.ndata['h'] = feats    
                weight = self.atom_weighting_specific[i](feats) 
                bg.ndata['w'] = weight  
                specific_feats_sum = sum_nodes(bg, 'h', 'w')  
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)

        # cal shared feats  
        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')
        # feat_list.append(shared_feats_sum)
        if self.attention:
            if self.return_weight:  
                return feat_list, atom_list 
            else:
                return feat_list
        else:
            return shared_feats_sum 

    def atom_weight(self, in_feats):
        return nn.Sequential(  
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )
       


class RGCNLayer(nn.Module):  
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()
        
        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=None, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual  
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats) 

        self.bn = batchnorm  
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        """Update atom representations
        Parameters
        ----------
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        node_feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        etype: int
            bond type
        norm: torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm) 
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats)) 
            new_feats = new_feats + res_feats  
        if self.bn:
            new_feats = self.bn_layer(new_feats)  
            
        del res_feats
        torch.cuda.empty_cache()  
        return new_feats
    
class FPN(nn.Module):
    def __init__(self, descriptor_dim, fp_2_dim, hidden_size, dropout, device):
        super(FPN, self).__init__()
        self.fp_dim = descriptor_dim 
        self.fp_2_dim = fp_2_dim      
        self.hidden_dim = hidden_size 
        self.dropout_fpn = dropout    
        self.device = device             

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, descriptor):     
        descriptor = torch.tensor(descriptor)  
        descriptor = descriptor.float().to(self.device)

        fpn_out = self.fc1(descriptor)         
        fpn_out = self.dropout(fpn_out)  
        fpn_out = self.act_func(fpn_out)  
        fpn_out = self.fc2(fpn_out)  

        return fpn_out
    

class BaseGNN(nn.Module):  
    def __init__(self, gnn_out_feats, descriptor, descriptor_dim, n_tasks, fpn_out, fp_2_dim, hidden_size, select_task_list, device, rgcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks 
        self.gnn_layers = nn.ModuleList()   
        self.return_weight = return_weight 
        self.FPN = FPN(descriptor_dim,fp_2_dim,hidden_size,dropout,device)

        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)  
        self.fc_in_feats = gnn_out_feats + 256
        
        
        self.return_mol_embedding=return_mol_embedding  
        
        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        
        self.output_layer1 = nn.ModuleList()
        
        for task in select_task_list:
            if task == 'toxicity_type_class':
                self.output_layer1.append(self.output_layer(classifier_hidden_feats, 6))  
            elif task == 'neurotoxicity_type_class':
                self.output_layer1.append(self.output_layer(classifier_hidden_feats, 4))  
            else:
                self.output_layer1.append(self.output_layer(classifier_hidden_feats, 1))  
                                 

    def forward(self, bg, node_feats, etype, descriptor, norm=None):
        # Update atom features with GNNs
        for gnn in self.gnn_layers:     
            node_feats = gnn(bg, node_feats, etype, norm)

        # Compute molecule features from atom features
        if self.return_weight:    
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)
            
        fpn_out = self.FPN(descriptor) 
               
        prediction_all = {}  
       
        for i in range(self.task_num):
            mol_feats = torch.cat([feats_list[i] * 0.2, fpn_out * 0.8], dim=1)
            
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)       
            prediction_all[f"task_{i}"] = predict            
                   
                
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return feats_list[0]
        else:
            # generate atom weight and atom feats
            if self.return_weight:
                return prediction_all, atom_weight_list, node_feats
            # just generate prediction
            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats) 
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


class MGA(BaseGNN):
    def __init__(self, in_feats, descriptor, descriptor_dim, rgcn_hidden_feats, n_tasks, fpn_out, fp_2_dim, hidden_size, select_task_list, device, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.0):
        super(MGA, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1], 
                                  descriptor=descriptor,
                                  descriptor_dim=descriptor_dim,
                                  n_tasks=n_tasks,                       
                                  fpn_out=fpn_out,
                                  fp_2_dim=fp_2_dim,
                                  hidden_size=hidden_size,
                                  device=device,
                                  select_task_list=select_task_list,
                                  classifier_hidden_feats=classifier_hidden_feats,
                                  return_mol_embedding=return_mol_embedding,
                                  return_weight=return_weight,
                                  rgcn_drop_out=rgcn_drop_out,
                                  dropout=dropout                                                              
                                  )
        
        
        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i] 
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set):
    smiles, graphs, descriptor, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels) 
    num_pos = 0
    num_impos = 0
    for i in labels[:, 0]:
        if i == 1:
            num_pos = num_pos + 1
        if i == 0:
            num_impos = num_impos + 1
    weight = num_impos / (num_pos+0.00000001)
    task_pos_weight = torch.tensor(weight) 
    return task_pos_weight

def multi_weight_six(train_set):  
    smiles, graphs, descriptor, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)

    class_counts = [0] * 6 
    for i in labels[:, 1]:
        if i == 0:
            class_counts[0] += 1
        elif i == 1:
            class_counts[1] += 1
        elif i == 2:
            class_counts[2] += 1
        elif i == 3:
            class_counts[3] += 1
        elif i == 4: 
            class_counts[4] += 1
        elif i == 5: 
            class_counts[5] += 1
            
    total_samples = sum(class_counts)
    class_weights = [total_samples / (count + 1e-8) for count in class_counts]
    return class_weights


def multi_weight_four(train_set):  
    smiles, graphs, descriptor, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)

    class_counts = [0] * 4 
    for i in labels[:, 2]:
        if i == 0:
            class_counts[0] += 1
        elif i == 1:
            class_counts[1] += 1
        elif i == 2:
            class_counts[2] += 1
        elif i == 3:
            class_counts[3] += 1

    total_samples = sum(class_counts)
    class_weights = [total_samples / (count + 1e-8) for count in class_counts]
    return class_weights


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask_c = [] 
        self.mask_r = []
        self.y_pred_c = {task_name: [] for task_name in ['task_0', 'task_1', 'task_2']} 
        self.y_pred_r = []
        self.y_pred_r = []
        self.y_true_c = []
        self.y_true_r = []

    def update_c(self, y_pred, y_true, mask):  
        """Update for the result of an iteration
        Parameters
        ----------        
        y_pred : dict
            Predicted molecule labels with keys corresponding to tasks,
            e.g., {'task_0': tensor(...), 'task_1': tensor(...)}
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """ 
        for task_name, pred in y_pred.items():
            self.y_pred_c[task_name].append(pred.detach().cpu()) #32 
            
        self.y_true_c.append(y_true.detach().cpu())  #32
        self.mask_c.append(mask.detach().cpu())   #32
        
    def update_r(self, y_pred, y_true, mask):     
        self.y_pred_r.append(y_pred.detach().cpu())  #32
        self.y_true_r.append(y_true.detach().cpu())  #32
        self.mask_r.append(mask.detach().cpu())  #32

    def activate_result(self,y_pred):
        processed_preds = {}
        for task_name, preds in y_pred.items():
            if task_name == 'task_0':
                processed_preds[task_name] = torch.sigmoid(preds)
            else:
                processed_preds[task_name] = torch.softmax(preds, dim=1)
        return processed_preds

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        mask = torch.cat(self.mask_c, dim=0)           
        y_true = torch.cat(self.y_true_c, dim=0) 
        # y_pred = torch.cat(self.y_pred_c, dim=0) 
        cat_preds = {}
        for task_name, preds in self.y_pred_c.items():
            if isinstance(preds, list): 
                task_preds = torch.cat(preds, dim=0) 
            else:
                task_preds = preds
            cat_preds[task_name] = task_preds  
        y_pred = cat_preds
        y_pred = self.activate_result(y_pred)
            
        n_tasks = y_true.shape[1]  
             
        scores = []
        for task in range(3):
            task_w = mask[:, task]         
            task_y_true = y_true[:, task][task_w != 0].numpy()           
            task_name = f'task_{task}'
            task_y_pred = y_pred[task_name][task_w != 0].numpy() 
            
            if task == 0:                  
                scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))       
        
            elif task == 1: 
                task_y_true1 = task_y_true.flatten()
                task_y_pred1 = task_y_pred.tolist()  
                
                df = pd.DataFrame({
                    'True Labels': task_y_true1,
                    'Predicted Values': task_y_pred1
                 })
                df.to_csv('Multi_six_1.csv', index=False)  
                scores.append(round(roc_auc_score(task_y_true, task_y_pred, multi_class='ovr'), 4))
              
            elif task == 2: 
                task_y_true1 = task_y_true.flatten()
                task_y_pred1 = task_y_pred.tolist() 

                df = pd.DataFrame({
                    'True Labels': task_y_true1,
                    'Predicted Values': task_y_pred1
                 })
                df.to_csv('Multi_four_1.csv', index=False)  
                                                     
                scores.append(round(roc_auc_score(task_y_true, task_y_pred, multi_class='ovr'), 4))
        return scores  

    def accuracy_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        mask = torch.cat(self.mask_c, dim=0)           
        y_true = torch.cat(self.y_true_c, dim=0) 
        # y_pred = torch.cat(self.y_pred_c, dim=0) 
        cat_preds = {}
        for task_name, preds in self.y_pred_c.items():
            if isinstance(preds, list): 
                task_preds = torch.cat(preds, dim=0) 
            else:
                task_preds = preds
            cat_preds[task_name] = task_preds  
        y_pred = cat_preds
        y_pred = self.activate_result(y_pred)
            
        n_tasks = y_true.shape[1]  
             
        scores = []
        for task in range(3):
            task_w = mask[:, task]         
            task_y_true = y_true[:, task][task_w != 0].numpy()           
            task_name = f'task_{task}'
            task_y_pred = y_pred[task_name][task_w != 0].numpy() 
            
            if task == 0:  
                scores.append(round(accuracy_score(task_y_true, task_y_pred.round()), 4))
            else:  
                scores.append(round(accuracy_score(task_y_true, task_y_pred.argmax(axis=1)), 4))
        return scores     
    
    
    def recall_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        mask = torch.cat(self.mask_c, dim=0)           
        y_true = torch.cat(self.y_true_c, dim=0) 
        # y_pred = torch.cat(self.y_pred_c, dim=0) 
        cat_preds = {}
        for task_name, preds in self.y_pred_c.items():
            if isinstance(preds, list): 
                task_preds = torch.cat(preds, dim=0) 
            else:
                task_preds = preds
            cat_preds[task_name] = task_preds  
        y_pred = cat_preds
        y_pred = self.activate_result(y_pred)
            
        n_tasks = y_true.shape[1]  
             
        scores = []
        for task in range(3):
            task_w = mask[:, task]         
            task_y_true = y_true[:, task][task_w != 0].numpy()           
            task_name = f'task_{task}'
            task_y_pred = y_pred[task_name][task_w != 0].numpy() 
            
            if task == 0: 
                scores.append(round(recall_score(task_y_true, task_y_pred.round(),average='binary'), 4))
            else: 
                scores.append(round(recall_score(task_y_true, task_y_pred.argmax(axis=1), average='macro'), 4))
        return scores      
    
    def precision_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        mask = torch.cat(self.mask_c, dim=0)           
        y_true = torch.cat(self.y_true_c, dim=0) 
        # y_pred = torch.cat(self.y_pred_c, dim=0) 
        cat_preds = {}
        for task_name, preds in self.y_pred_c.items():
            if isinstance(preds, list): 
                task_preds = torch.cat(preds, dim=0) 
            else:
                task_preds = preds
            cat_preds[task_name] = task_preds  
        y_pred = cat_preds
        y_pred = self.activate_result(y_pred)
            
        n_tasks = y_true.shape[1]  #获取任务数
             
        scores = []
        for task in range(3):
            task_w = mask[:, task]         
            task_y_true = y_true[:, task][task_w != 0].numpy()           
            task_name = f'task_{task}'
            task_y_pred = y_pred[task_name][task_w != 0].numpy() 
            
            if task == 0:  
                scores.append(round(precision_score(task_y_true, task_y_pred.round(), average='binary'), 4))
            else:  
                scores.append(round(precision_score(task_y_true, task_y_pred.argmax(axis=1), average='macro'), 4))
        return scores      

    def f1_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """

        mask = torch.cat(self.mask_c, dim=0)           
        y_true = torch.cat(self.y_true_c, dim=0) 
        # y_pred = torch.cat(self.y_pred_c, dim=0) 
        cat_preds = {}
        for task_name, preds in self.y_pred_c.items():
            if isinstance(preds, list): 
                task_preds = torch.cat(preds, dim=0) 
            else:
                task_preds = preds
            cat_preds[task_name] = task_preds  
        y_pred = cat_preds
        y_pred = self.activate_result(y_pred)
            
        n_tasks = y_true.shape[1]  
             
        scores = []
        for task in range(3):
            task_w = mask[:, task]         
            task_y_true = y_true[:, task][task_w != 0].numpy()           
            task_name = f'task_{task}'
            task_y_pred = y_pred[task_name][task_w != 0].numpy() 
            
            if task == 0:  
                scores.append(round(f1_score(task_y_true, task_y_pred.round(), average='binary'), 4))
            else:  
                scores.append(round(f1_score(task_y_true, task_y_pred.argmax(axis=1), average='macro'), 4))
        return scores          
    

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask_c, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask_r, dim=0)
        y_pred = torch.cat(self.y_pred_r, dim=0)
        y_true = torch.cat(self.y_true_r, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            mse = F.mse_loss(task_y_pred, task_y_true)
            rmse = torch.sqrt(mse)
            scores.append(rmse.item())
#             scores.append(torch.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores
    
    def mse(self):
        """Compute MSE for each task.
        Returns
        -------
        list of float
            mse for all tasks
        """
        mask = torch.cat(self.mask_r, dim=0)
        y_pred = torch.cat(self.y_pred_r, dim=0)
        y_true = torch.cat(self.y_true_r, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            mse = F.mse_loss(task_y_pred, task_y_true).cpu().item()
            scores.append(mse)

        return scores


    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask_r, dim=0)
        y_pred = torch.cat(self.y_pred_r, dim=0)
        y_true = torch.cat(self.y_true_r, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask_r, dim=0)
        y_pred = torch.cat(self.y_pred_r, dim=0)
        y_true = torch.cat(self.y_true_r, dim=0)

        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_names, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert isinstance(metric_names, list), "metric_names should be a list of metric names."
        results = []
        for metric_name in metric_names:
            assert metric_name in ['roc_auc', 'l1', 'rmse', 'mse', 'mae', 'roc_prc', 'r2', 'return_pred_true', 'accuracy', 'precision', 'recall', 'f1'], \
            f'Expect metric name to be one of "roc_auc", "l1", "rmse", "mse", "mae", "roc_prc", "r2", "return_pred_true", "accuracy", "precision", "recall", "f1", got {metric_name}'

            assert reduction in ['mean', 'sum'], f"Expected reduction to be 'mean' or 'sum', got {reduction}"

            # 根据 metric_name 调用相应的方法
            if metric_name == 'roc_auc':
                results.append(self.roc_auc_score())
            elif metric_name == 'l1':
                results.append(self.l1_loss(reduction))
            elif metric_name == 'rmse':
                results.append(self.rmse())
            elif metric_name == 'mae':
                results.append(self.mae())
            elif metric_name == 'mse':
                results.append(self.mse())
            elif metric_name == 'roc_prc':
                results.append(self.roc_precision_recall_score())
            elif metric_name == 'r2':
                results.append(self.r2())
            elif metric_name == 'return_pred_true':
                results.append(self.return_pred_true())
            elif metric_name == 'accuracy':
                results.append(self.accuracy_score())
            elif metric_name == 'precision':
                results.append(self.precision_score())
            elif metric_name == 'recall':
                results.append(self.recall_score())
            elif metric_name == 'f1':
                results.append(self.f1_score())
        return results


def collate_molgraphs(data):  
    smiles, graphs, descriptor,labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)  
    bg.set_e_initializer(dgl.init.zero_initializer)  
    labels = torch.tensor(labels)
    mask = torch.tensor(mask)
    descriptor = [np.array(ast.literal_eval(f)) for f in descriptor]
    
    def normalization(data):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data  
        
    descriptor = normalization(descriptor)

    return smiles, bg, descriptor, labels, mask
    
def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c_0,loss_criterion_c_1,loss_criterion_c_2, loss_criterion_r, optimizer, task_weight=None):
    model.train()  
    train_meter_c = Meter() 
    train_meter_r = Meter() 
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])

    for batch_id, batch_data in enumerate(data_loader): 
        smiles, bg,descriptor, labels, mask = batch_data
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])  
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
       # descriptor = descriptor.float().to(args['device']) 
        
        logits = model(bg, atom_feats, bond_feats, descriptor, norm=None)  
       # print(f"logits: {logits}")
        task_0_pred = logits["task_0"]
        labels = labels.type_as(task_0_pred).to(args['device']) 
    
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            task_names = ['task_0', 'task_1', 'task_2']
            logits_c = {task_name: logits[task_name] for task_name in task_names}
            
            labels_c = labels[:,:args['classification_num']]            
            mask_c = mask[:,:args['classification_num']]       
            logits_r = logits["task_3"]
            labels_r = labels[:,args['classification_num']:] 
            
            mask_r = mask[:,args['classification_num']:]         
 
            
            # chose loss function according to task_weight  
            if task_weight is None:                                                
                loss_c_0 = loss_criterion_c_0(logits_c["task_0"], labels_c[:, 0].unsqueeze(1)) * (mask_c[:, 0].unsqueeze(1) != 0).float()
                
                ignore_index = 123456
                loss_criterion_c_1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_1 = labels_c[:, 1]
                mask_invalid = torch.isin(labels_c_1, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_1.device))
                labels_c_1 = labels_c_1.masked_fill(mask_invalid, ignore_index)
                loss_c_1 = loss_criterion_c_1(logits_c["task_1"], labels_c_1.long())
                

                loss_criterion_c_2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_2 = labels_c[:, 2]
                mask_invalid = torch.isin(labels_c_2, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_2.device))
                labels_c_2 = labels_c_2.masked_fill(mask_invalid, ignore_index)
                loss_c_2 = loss_criterion_c_2(logits_c["task_2"], labels_c_2.long())           
                
                loss_r = loss_criterion_r(logits_r, labels_r) * (mask_r != 0).float()  
                loss = (loss_c_0 + loss_c_1 + loss_c_2).mean()+ loss_r.mean()
                
            else:
                task_weight_c = task_weight[:args['classification_num']]
                task_weight_r = task_weight[args['classification_num']:] 
                         
                loss_c_0 = loss_criterion_c_0(logits_c["task_0"], labels_c[:, 0].unsqueeze(1)) * (mask_c[:, 0].unsqueeze(1) != 0).float()

                ignore_index = 123456
                loss_criterion_c_1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_1 = labels_c[:, 1]
                mask_invalid = torch.isin(labels_c_1, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_1.device))
                labels_c_1 = labels_c_1.masked_fill(mask_invalid, ignore_index)
                loss_c_1 = torch.mean(loss_criterion_c_1(logits_c["task_1"], labels_c_1.long()), dim=0)*task_weight_c
                            
                loss_criterion_c_2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_2 = labels_c[:, 2]
                mask_invalid = torch.isin(labels_c_2, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_2.device))
                labels_c_2 = labels_c_2.masked_fill(mask_invalid, ignore_index)
                loss_c_2 = torch.mean(loss_criterion_c_2(logits_c["task_2"], labels_c_2.long()), dim=0)*task_weight_c   
                
                loss_r = (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()       
                loss = (loss_c_0 + loss_c_1 + loss_c_2).mean()+ loss_r

              
            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()     
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
  
            train_meter_c.update_c(logits_c, labels_c, mask_c)  
            train_meter_r.update_r(logits_r, labels_r, mask_r)
            del bg, mask, labels, atom_feats, bond_feats, loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
            torch.cuda.empty_cache()
            
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            if task_weight is None:
                loss_c_0 = loss_criterion_c_0(logits_c["task_0"], labels_c[:, 0].unsqueeze(1)) * (mask_c[:, 0].unsqueeze(1) != 0).float()
                
                ignore_index = 123456
                loss_criterion_c_1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_1 = labels_c[:, 1]

                mask_invalid = torch.isin(labels_c_1, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_1.device))
                labels_c_1 = labels_c_1.masked_fill(mask_invalid, ignore_index)
                loss_c_1 = loss_criterion_c_1(logits_c["task_1"], labels_c_1.long())                

                loss_criterion_c_2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_2 = labels_c[:, 2]
                mask_invalid = torch.isin(labels_c_2, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_2.device))
                labels_c_2 = labels_c_2.masked_fill(mask_invalid, ignore_index)
                loss_c_2 = loss_criterion_c_2(logits_c["task_2"], labels_c_2.long())            
                
                loss = (loss_c_0 + loss_c_1 + loss_c_2).mean()+ loss_r.mean()                               

                
            else:         
                loss_c_0 = loss_criterion_c_0(logits_c["task_0"], labels_c[:, 0].unsqueeze(1)) * (mask_c[:, 0].unsqueeze(1) != 0).float()
                
                ignore_index = 123456
                loss_criterion_c_1 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_1 = labels_c[:, 1]
                mask_invalid = torch.isin(labels_c_1, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_1.device))

                labels_c_1 = labels_c_1.masked_fill(mask_invalid, ignore_index)
                loss_c_1 = torch.mean(loss_criterion_c_1(logits_c["task_1"], labels_c_1.long()), dim=0)*task_weight_c
                            
                loss_criterion_c_2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
                labels_c_2 = labels_c[:, 2]
                mask_invalid = torch.isin(labels_c_2, torch.tensor([ignore_index], dtype=torch.long, device=labels_c_2.device))
                labels_c_2 = labels_c_2.masked_fill(mask_invalid, ignore_index)
                loss_c_2 = torch.mean(loss_criterion_c_2(logits_c["task_2"], labels_c_2.long()), dim=0)*task_weight_c   
                     
                loss = (loss_c_0 + loss_c_1 + loss_c_2).mean()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_c.update_c(logits_c, labels_c, mask_c)
            del bg, mask, labels, atom_feats, bond_feats, loss,  logits
            torch.cuda.empty_cache()
            
        else:
            # chose loss function according to task_weight
            if task_weight is None:
              #  loss_r = (loss_criterion_r(logits_r, labels_r) * (mask_r != 0).float()).mean()               
                loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            else:
               # loss_r = (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()   
                loss = (torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
            #     epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
            train_meter_r.update_c(logits, labels, mask)
            del bg, mask, labels, atom_feats, bond_feats, loss, logits
            torch.cuda.empty_cache()
            
    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], 'r2+auc', train_score))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['regression_metric_name'], train_score))


def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg,descriptor, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
          #  descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(bg, atom_feats, bond_feats,descriptor, norm=None)
            
            task_0_pred = logits["task_0"]
            labels = labels.type_as(task_0_pred).to(args['device'])
            
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                task_names = ['task_0', 'task_1', 'task_2']
                logits_c = {task_name: logits[task_name] for task_name in task_names}

                labels_c = labels[:,:args['classification_num']]
                mask_c = mask[:,:args['classification_num']]       
                logits_r = logits["task_3"]
                labels_r = labels[:,args['classification_num']:] 
                mask_r = mask[:,args['classification_num']:]   
            
                # Mask non-existing labels
                eval_meter_c.update_c(logits_c, labels_c, mask_c)
                eval_meter_r.update_r(logits_r, labels_r, mask_r)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r
                torch.cuda.empty_cache()
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update_c(logits, labels, mask)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
            else:
                # Mask non-existing labels
                eval_meter_r.update_r(logits, labels, mask)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            return eval_meter_c.compute_metric(args['classification_metric_name'])
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])
        

def run_an_eval_epoch_heterogeneous_predict(args, model, data_loader):
    model.eval()  
    with torch.no_grad():  
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, descriptor, = batch_data  
            # descriptor = descriptor.float().to(args['device'])  
            
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])        
            logits = model(bg, atom_feats, bond_feats, descriptor, norm=None)
            
            print(f"Prediction for batch {batch_id}:")
            for task_name, task_pred in logits.items():
                print(f"  Task {task_name}: {task_pred.cpu().numpy()}") 

        
        
        
def run_an_eval_epoch_pih(args, model, data_loader, output_path):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    smiles_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, descriptor,labels, mask = batch_data
            smiles_list = smiles_list + smiles
            labels = labels.float().to(args['device'])
            mask = mask.float().to(args['device'])
          #  descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(bg, atom_feats, bond_feats,descriptor, norm=None)
            task_0_pred = logits["task_0"]
            labels = labels.type_as(task_0_pred).to(args['device']) 
            
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                
                task_names = ['task_0', 'task_1', 'task_2']
                logits_c = {task_name: logits[task_name] for task_name in task_names}
                labels_c = labels[:,:args['classification_num']]
                mask_c = mask[:,:args['classification_num']]       
                logits_r = logits["task_3"]
                labels_r = labels[:,args['classification_num']:] 
                mask_r = mask[:,args['classification_num']:]
            
                # Mask non-existing labels
                eval_meter_c.update_c(logits_c, labels_c, mask_c)
                eval_meter_r.update_r(logits_r, labels_r, mask_r)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r                                
                torch.cuda.empty_cache()
                
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update_c(logits, labels, mask)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()                
            else:
                # Mask non-existing labels
                eval_meter_r.update_r(logits, labels, mask)
                del smiles, bg, mask, labels, atom_feats, bond_feats, logits
                torch.cuda.empty_cache()
        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            y_pred, y_true = eval_meter_c.compute_metric('return_pred_true')
            result = pd.DataFrame(columns=['smiles', 'pred', 'true'])
            result['smiles'] = smiles_list
            result['pred'] = np.squeeze(y_pred.numpy()).tolist()
            result['true'] = np.squeeze(y_true.numpy()).tolist()
            result.to_csv(output_path, index=None)
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])


def run_an_eval_epoch_heterogeneous_return_weight(args, model, data_loader, vis_list=None, vis_task='CYP2D6'):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, descriptor,labels, mask = batch_data
            #####
            labels = labels.float().to(args['device'])
          #  descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list, node_feats,descriptor = model(bg, atom_feats, bond_feats,descriptor, norm=None)
            
            task_0_pred = logits["task_0"]
            labels = labels.type_as(task_0_pred).to(args['device']) 
            
            task_names = ['task_0', 'task_1', 'task_2']
            logits_c = {task_name: logits[task_name] for task_name in task_names}

            logits_c = torch.sigmoid(logits_c)
            # different tasks with different atom weight

            for mol_index in range(len(smiles)):
                atom_smiles = smiles[mol_index]
                if atom_smiles in vis_list:
                    for tasks_index in range(31):
                        # if args['all_task_list'][tasks_index] == vis_task:
                        if labels[mol_index, tasks_index]!=123456:
                            bg.ndata['w'] = atom_weight_list[tasks_index]
                            bg.ndata['feats'] = node_feats
                            unbatch_bg = dgl.unbatch(bg)
                            one_atom_weight = unbatch_bg[mol_index].ndata['w']
                            one_atom_feats = unbatch_bg[mol_index].ndata['feats']
                            # visual selected molecules
                            print('Tasks:', tasks_index, args['all_task_list'][tasks_index], "**********************")
                            if tasks_index < 26:
                                print('Predict values:', logits_c[mol_index, tasks_index])
                            else:
                                print('Predict values:', logits[mol_index, tasks_index])
                            print('True values:', labels[mol_index, tasks_index])
                            weight_visualization.weight_visulize(atom_smiles, one_atom_weight)
                else:
                    continue


def run_an_eval_epoch_heterogeneous_return_weight_py(args, model, data_loader, vis_list=None, vis_task='CYP2D6'):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg,descriptor, labels, mask = batch_data
            #####
            labels = labels.float().to(args['device'])
           # descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            print(bond_feats)
            logits, atom_weight_list, node_feats = model(bg, atom_feats, bond_feats,descriptor, norm=None)
            
            task_0_pred = logits["task_0"]
            labels = labels.type_as(task_0_pred).to(args['device']) 
            
            task_names = ['task_0', 'task_1', 'task_2']
            logits_c = {task_name: logits[task_name] for task_name in task_names}
            logits_c = torch.sigmoid(logits_c)
            
            # different tasks with different atom weight

            for mol_index in range(len(smiles)):
                atom_smiles = smiles[mol_index]
                if atom_smiles in vis_list:
                    for tasks_index in range(31):
                        if args['all_task_list'][tasks_index] == vis_task:
                            if labels[mol_index, tasks_index]!=123456:
                                bg.ndata['w'] = atom_weight_list[tasks_index]
                                bg.ndata['feats'] = node_feats
                                unbatch_bg = dgl.unbatch(bg)
                                one_atom_weight = unbatch_bg[mol_index].ndata['w']
                                one_atom_feats = unbatch_bg[mol_index].ndata['feats']
                                # visual selected molecules
                                print('Tasks:', tasks_index, args['all_task_list'][tasks_index], "**********************")
                                if tasks_index < 26:
                                    print('Predict values:', logits_c[mol_index, tasks_index])
                                else:
                                    print('Predict values:', logits[mol_index, tasks_index])
                                print('True values:', labels[mol_index, tasks_index])
                                weight_visualization.weight_visulize_py(atom_smiles, one_atom_weight)
                else:
                    continue


def run_an_eval_epoch_heterogeneous_generate_weight(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id+1, len(data_loader)))
            smiles, bg, labels, mask, descriptor = batch_data
            labels = labels.float().to(args['device'])
       #     descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats,descriptor, norm=None)
            for atom_weight in atom_weight_list:
                atom_list_all.append(atom_weight[args['select_task_index']])
    task_name = args['select_task_list'][0]
    atom_weight_list = pd.DataFrame(atom_list_all, columns=['atom_weight'])
    atom_weight_list.to_csv(task_name+"_atom_weight.csv", index=None)


def generate_chemical_environment(args, model, data_loader):  
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id + 1, len(data_loader)))
            smiles, bg, labels, mask, descriptor = batch_data
            print(bg.ndata[args['atom_data_field']][1])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits, atom_weight_list = model(bg, atom_feats, bond_feats, norm=None)
            print('after training:', bg.ndata['h'][1])


def generate_mol_feats(args, model, data_loader, dataset_output_path):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg,descriptor, labels, mask = batch_data
         #   descriptor = descriptor.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            feats = model(bg, atom_feats, bond_feats,descriptor, norm=None).numpy().tolist()
            feats_name = ['graph-feature' + str(i+1) for i in range(64)]
            data = pd.DataFrame(feats, columns=feats_name)
            data['smiles'] = smiles
            data['descriptor'] = descriptor
            data['labels'] = labels.squeeze().numpy().tolist()
    data.to_csv(dataset_output_path, index=None)


class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='model/{}_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        # model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        if torch.cuda.is_available():
            pretrained_model = torch.load('model/'+self.pretrained_model)
        else:
            pretrained_model = torch.load('model/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        if torch.cuda.is_available():
            pretrained_model = torch.load('model/' + self.pretrained_model)
        else:
            pretrained_model = torch.load('model/' + self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)


