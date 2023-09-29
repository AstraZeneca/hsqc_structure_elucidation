import torch
import torch.nn as nn

import dgl
from dgl.nn.pytorch import NNConv, Set2Set


class MLPNodeReadout(nn.Module):

    def __init__(self, node_feats, graph_feats):
        super(MLPNodeReadout, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(node_feats, graph_feats), nn.ReLU(),
            nn.Linear(graph_feats, graph_feats), nn.ReLU(),
            nn.Linear(graph_feats, graph_feats), nn.ReLU(),
            nn.Linear(graph_feats, graph_feats), nn.ReLU()
        )

    def forward(self, g, node_feats):

        node_feats = self.project(node_feats)
       
        with g.local_scope():
            g.ndata['h'] = node_feats
            graph_feats = dgl.sum_nodes(g, 'h')

        return graph_feats
        
        
class nmr_mpnn_BASELINE(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, readout_mode, 
                 node_feats, node_hid_feats, pred_hid_feats,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 prob_dropout = 0.1):
        
        super(nmr_mpnn_BASELINE, self).__init__()

        self.readout_mode = readout_mode
        
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_feats), nn.Tanh(),
        )

        
        self.num_step_message_passing = num_step_message_passing
        

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_feats * node_feats), nn.ReLU()
        )

        
        self.gnn_layer = NNConv(
            in_feats = node_feats,
            out_feats = node_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.gru = nn.GRU(node_feats, node_feats)
        
        
        if self.readout_mode == 'proposed_set2set':
            # set2setReadout
            self.readout_g = Set2Set(input_dim = node_feats + node_in_feats,
                                n_iters = num_step_set2set,
                                n_layers = num_layer_set2set)
            
            self.readout_n = nn.Sequential(
                nn.Linear(node_feats * (1 + num_step_message_passing) + (node_feats + node_in_feats) * 2, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, 1)
            )    

        elif self.readout_mode == 'proposed_mlp':      
            # MLPNodeReadout
            self.readout_g = MLPNodeReadout(node_feats + node_in_feats, pred_hid_feats)
                        
            self.readout_n = nn.Sequential(
                nn.Linear(node_feats * (1 + num_step_message_passing) + pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, 1)
            )   
        


        self.readout_n_naive = nn.Sequential(
            nn.Linear(node_feats * (1 + num_step_message_passing), pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )                   

    def forward(self, g, n_nodes, masks):
        
        def embed(g):
            
            node_feats_in = g.ndata['node_attr']
            node_feats = self.project_node_feats(node_feats_in)

            edge_feats = g.edata['edge_attr']
            
            node_aggr1 = [node_feats]
            node_aggr2 = [node_feats_in]
            for _ in range(self.num_step_message_passing):
                msg = self.gnn_layer(g, node_feats, edge_feats).unsqueeze(0)
                _, node_feats = self.gru(msg, node_feats.unsqueeze(0))
                node_feats = node_feats.squeeze(0)
                
                node_aggr1.append(node_feats)
            
            node_aggr2.append(node_feats)    
            
            node_aggr1 = torch.cat(node_aggr1, 1)
            node_aggr2 = torch.cat(node_aggr2, 1)
            
            return node_aggr1, node_aggr2

        node_embed_feats, node_embed_feats2 = embed(g)

        if self.readout_mode == 'baseline':
            out = self.readout_n_naive(node_embed_feats[masks])


        elif self.readout_mode in ['proposed_set2set', 'proposed_mlp']:
            graph_embed_feats = self.readout_g(g, node_embed_feats2)        
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

            out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[masks])
        

        return out[:,0]
