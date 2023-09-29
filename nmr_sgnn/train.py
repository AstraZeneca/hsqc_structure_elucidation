import numpy as np

import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import RandomSplitter

from dataset import GraphDataset
from util import collate_reaction_graphs
from model import training, inference

from gnn_models.mpnn_proposed import nmr_mpnn_PROPOSED
from gnn_models.mpnn_baseline import nmr_mpnn_BASELINE

from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


def train(args):
    target = args.target
    message_passing_mode = args.message_passing_mode
    readout_mode = args.readout_mode
    graph_representation = args.graph_representation
    memo = args.memo
    fold_seed = args.fold_seed

    node_embedding_dim = args.node_embedding_dim
    node_hidden_dim = args.node_hidden_dim
    readout_n_hidden_dim = args.readout_n_hidden_dim

    data_split = [0.95, 0.05]
    batch_size = 128
    
    use_training = False

    if not os.path.exists('./model'):
        os.mkdir('./model')
    
    if memo:
        model_path = f'./model/{target}_{graph_representation}_{message_passing_mode}_{readout_mode}_{fold_seed}_{memo}.pt'
    else:
        model_path = f'./model/{target}_{graph_representation}_{message_passing_mode}_{readout_mode}_{fold_seed}.pt'

    
    random_seed = 27407 + fold_seed

    data = GraphDataset(target, graph_representation)
    kfold = RandomSplitter()
    data_fold = kfold.k_fold_split(data, k=10, random_state=27407, log=False)
    trainval_set, test_set = data_fold[fold_seed]

    train_set, val_set = split_dataset(trainval_set, data_split, shuffle=True, random_state=random_seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    
    train_y = np.hstack([inst[-2][inst[-1]] for inst in iter(train_loader.dataset)])
    train_y_mean, train_y_std = np.mean(train_y), np.std(train_y)

    node_dim = data.node_attr.shape[1]
    edge_dim = data.edge_attr.shape[1]

    if message_passing_mode == 'proposed':
        net = nmr_mpnn_PROPOSED(node_dim, edge_dim, readout_mode, node_embedding_dim, readout_n_hidden_dim).cuda()
    elif message_passing_mode == 'baseline':
        net = nmr_mpnn_BASELINE(node_dim, edge_dim, readout_mode, node_embedding_dim, node_hidden_dim, readout_n_hidden_dim).cuda()
        
    

    print('--- data_size:', trainval_set.__len__())
    print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('--- model_path:', model_path)



    if use_training:
        print('-- Load Trained model')
        net.load_state_dict(torch.load(model_path))

    else:
        # training
        print('-- TRAINING')
        net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path)
    
    # inference
    test_y_pred, time_per_mol = inference(net, test_loader, train_y_mean, train_y_std)
    
    if target == '13C':
        test_y = np.hstack([inst[-3][inst[-1]] for inst in iter(test_loader.dataset)])
        test_mae = mean_absolute_error(test_y, test_y_pred)
        test_rmse = mean_squared_error(test_y, test_y_pred)**0.5
    
    elif target == '1H':
        test_y = np.hstack([inst[-3] for inst in iter(test_loader.dataset)])
        test_numHs = np.hstack([inst[-4] for inst in iter(test_loader.dataset)])
        test_y_pred2 = np.repeat(test_y_pred, test_numHs)
        
        test_mae = mean_absolute_error(test_y, test_y_pred2)
        test_rmse = mean_squared_error(test_y, test_y_pred2)**0.5

    print('-- prediction RESULT')
    print('--- test MAE      ', test_mae)
    print('--- test RMSE     ', test_rmse)

    return net, test_mae, test_rmse, time_per_mol