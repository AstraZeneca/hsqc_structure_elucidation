# This code is based on the implementation of Path-Augmented Graph Transformer Network in DGL-LifeSci
# https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/pagtn.py

import torch
import torch.nn as nn

from dgl.nn.pytorch import Set2Set

from dgl.nn.functional import edge_softmax
import dgl.function as fn
import dgl



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

class nmr_mpnn_PROPOSED(nn.Module):

    def __init__(self, node_in_feats, edge_feats, readout_mode, 
                 node_feats,
                 pred_hid_feats, 
                 depth=5,
                 n_heads=5,
                 dropout=0.1,
                 activation=nn.ReLU(),
                 prob_dropout = 0.1,
                 ):
        
        super(nmr_mpnn_PROPOSED, self).__init__()

        self.readout_mode = readout_mode

        
        self.gnn = PAGTNGNN(node_in_feats, node_feats, edge_feats,
                              depth, n_heads, activation)
        

        
        if self.readout_mode == 'proposed_set2set':
            self.readout_g = Set2Set(input_dim = node_feats + node_in_feats,
                                n_iters = 3,
                                n_layers = 1)
            
            self.readout_n = nn.Sequential(
                nn.Linear(node_feats * 3 + node_in_feats * 3, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, 1)
        )
        
        elif self.readout_mode == 'proposed_mlp':      
            # MLPNodeReadout
            self.readout_g = MLPNodeReadout(node_feats + node_in_feats, pred_hid_feats)
                        
            self.readout_n = nn.Sequential(
                nn.Linear(node_feats + node_in_feats + pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
                nn.Linear(pred_hid_feats, 1)
            )
        

                               

        self.readout_n_naive = nn.Sequential(
            nn.Linear(node_feats + node_in_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )       


    def forward(self, g, n_nodes, masks):
        
        def embed(g):
            
            node_feats = g.ndata['node_attr']
            edge_feats = g.edata['edge_attr']
            
            
            node_feats_embedding = self.gnn(g, node_feats, edge_feats)
            node_feats = torch.cat([node_feats_embedding, node_feats], dim=1)

            
            return node_feats
            
        node_embed_feats = embed(g)
        
        
        if self.readout_mode == 'baseline':
            out = self.readout_n_naive(node_embed_feats[masks])
        
        elif self.readout_mode in ['proposed_set2set', 'proposed_mlp']:
            graph_embed_feats = self.readout_g(g, node_embed_feats)      
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)
            out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[masks])
        
        return out[:,0]



class PAGTNLayer(nn.Module):
    """
    Single PAGTN layer from `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    edge_feats : int
        Size for the input edge features.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_feats,
                 activation=nn.ReLU()):
        super(PAGTNLayer, self).__init__()
        self.attn_src = nn.Linear(node_in_feats, node_in_feats)
        self.attn_dst = nn.Linear(node_in_feats, node_in_feats)
        self.attn_edg = nn.Linear(edge_feats, node_in_feats)
        self.attn_dot = nn.Linear(node_in_feats, 1)
        self.msg_src = nn.Linear(node_in_feats, node_out_feats)
        self.msg_dst = nn.Linear(node_in_feats, node_out_feats)
        self.msg_edg = nn.Linear(edge_feats, node_out_feats)
        self.wgt_n = nn.Linear(node_in_feats, node_out_feats)
        self.act = activation
        

    

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats) or (V, n_head, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats) or (V, n_head, node_out_feats)
            Updated node features.
        """

        g = g.local_var()
        # In the paper node_src, node_dst, edge feats are concatenated
        # and multiplied with the matrix. We have optimized this step
        # by having three separate matrix multiplication.
        
        g.ndata['src'] = self.attn_src(node_feats)
        g.ndata['dst'] = self.attn_dst(node_feats)
        edg_atn = self.attn_edg(edge_feats).unsqueeze(-2)

        g.apply_edges(fn.u_add_v('src', 'dst', 'e'))
        atn_scores = self.act(g.edata.pop('e') + edg_atn)

        atn_scores = self.attn_dot(atn_scores)
        atn_scores = edge_softmax(g, atn_scores)



        g.ndata['src'] = self.msg_src(node_feats)
        g.ndata['dst'] = self.msg_dst(node_feats)
        g.apply_edges(fn.copy_src('dst', 'e'))
        atn_inp = g.edata.pop('e') + self.msg_edg(edge_feats).unsqueeze(-2)
        
        g.edata['msg'] = atn_scores * atn_inp
        g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'feat'))
        out = g.ndata.pop('feat') + self.wgt_n(node_feats)
        
        return out


class PAGTNGNN(nn.Module):
    """Multilayer PAGTN model for updating node representations.
    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    node_hid_feats : int
        Size for the hidden node features.
    edge_feats : int
        Size for the input edge features.
    depth : int
        Number of PAGTN layers to be applied.
    nheads : int
        Number of attention heads.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """

    def __init__(self,
                 node_in_feats,
                 node_hid_feats,
                 edge_feats,
                 depth=5,
                 nheads=5,
                 activation=nn.ReLU()):
        super(PAGTNGNN, self).__init__()
        self.depth = depth
        self.nheads = nheads
        self.node_hid_feats = node_hid_feats

        self.atom_inp = nn.Linear(node_in_feats, node_hid_feats * nheads)
        
        self.model = nn.ModuleList([PAGTNLayer(node_hid_feats, node_hid_feats,
                                               edge_feats,
                                               activation)
                                    for _ in range(depth)])
        
        self.act = activation


    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node features.
        """
        g = g.local_var()
                
        atom_input = self.atom_inp(node_feats).view(-1, self.nheads, self.node_hid_feats)
        atom_input = self.act(atom_input)

        atom_h = atom_input
        for i in range(self.depth):
            attn_h = self.model[i](g, atom_h, edge_feats)            
            atom_h = torch.nn.functional.relu(attn_h + atom_input)
        
        atom_h = atom_h.mean(1)
        
        return atom_h