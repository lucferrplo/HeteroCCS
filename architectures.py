'''
CCS prediction using graph neural networks (GNNs) on
stereolectronics-infused molecular graphs (SIMGs).


References:

    - Large-scale prediction of collision cross-section 
    with very deep graph convolutional network for small molecule identification, 
    Xie et al. (https://doi.org/10.1016/j.chemolab.2024.105177)

    - Advancing molecular machine learning representations with stereoelectronics-infused molecular graphs, 
    Boiko et al. (https://doi.org/10.1038/s42256-025-01031-9)
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_geometric import nn as geom_nn
from torch_geometric.utils import softmax
from torch_scatter import scatter

import pytorch_lightning as pl









#######################################################################################
################################ GraphCCS (homogeneous) ###############################
#######################################################################################
class ScalingLayer(nn.Module):
    '''
    Learn a per-feature scaling vector to stabilize training in the deep architecture.
    Conceptually, instead of adding the full residual branch, the model adds a scaled version:
    h(l+1)=h(l)+ε⊙F(h(l)),
    where h(l) is the embedding of layer l and epsilon is the learned vector.

    
    Parameters
    ----------
    layer_dim : int
        Number of features.
    depth_idx : int
        Index of the layer in the concatenated architecture.
    '''
    def __init__(self, layer_dim, depth_idx):
        super().__init__()

        # for early layers increase epsilon to compensate for vanishing gradient
        if depth_idx <= 18:
            init_scale = 0.1
        elif depth_idx <= 24:
            init_scale = 1e-5
        else:
            init_scale = 1e-6

        self.scale = nn.Parameter(torch.full((layer_dim,), init_scale)) # learnable



    def forward(self, x):
        return x * self.scale






class ResAttGCN(nn.Module):
    '''
    Attentive graph convolutional neural network (GCN) with scaled residual connection.
    By design, the input and output dimensions have the same number of features.


    Parameters
    ----------
    GCdim : int
        Output feature dimension of the GCN.
    depth_idx : int
        Index of the layer in the concatenated architecture.
    dropout : float
    '''
    def __init__(self, GCdim, depth_idx, dropout):
        super().__init__()

        self.linear = nn.Linear(GCdim, GCdim)
        self.dropout = nn.Dropout(dropout)
        self.scaling_layer = ScalingLayer(GCdim, depth_idx)
    


    def forward(self, x, edge_index, edge_attr):
        # 1. graph attention
        # analogous to dgl.function.u_add_e
        # computes a message on an edge by performing element-wise add between features of u and e
        msgs = x[edge_index[0]] + edge_attr
        atts = softmax(msgs, edge_index[1], num_nodes=x.shape[0]) # turn the raw edge scores into attention weights with softmax
        # edge_index[1] tells which node each edge points to
        # softmax groups edges by their destination node and applies softmax separately within each group.
        # softmax is applied independently per feature?
        att_sum = scatter( # attention sum
            msgs * atts,
            edge_index[1],
            dim=0,
            dim_size=x.shape[0], # create output with size dim_size at dimension dim
            reduce="sum",
        ) # so the output is n_nodes x GCdim?


        # 2. feed forward linearly
        out = self.linear(att_sum)
        out = F.relu(out)
        out = self.dropout(out)


        # 3. scaled residual update
        out = self.scaling_layer(out) + x
        return out






class AttentiveFPReadout_step(nn.Module):
    '''
    One attentive read-out step.
    Code from: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/readout/attentivefp_readout.py


    Parameters
    ----------
    dimPool : int
        Number of features.
    dropout : float
    '''
    def __init__(self, dimPool, dropout):
        super().__init__()

        self.compute_logits = nn.Sequential(
            nn.Linear(2 * dimPool, 1),
            nn.LeakyReLU()
        )

        self.project_nodes = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dimPool, dimPool)
        )

        self.gru = nn.GRUCell(dimPool, dimPool)



    def forward(self, x, graph_embedding, batch):
        batch_size = graph_embedding.shape[0]

        # have the graph_embedding be repeated once per node
        # dgl.broadcast_nodes
        graph_embedding_per_node = F.relu(graph_embedding[batch]) # generate a node feature equal to the graph-level feature

        logits = self.compute_logits( # attention logits per node.
            torch.cat([graph_embedding_per_node, x], dim=1)
        )

        weights = softmax(logits, index = batch, num_nodes=batch_size)

        projected_nodes = self.project_nodes(x)

        add_pool = scatter( # weighted graph sum
            weights * projected_nodes,
            batch,
            dim=0,
            dim_size=batch_size, # create output with size dim_size at dimension dim
            reduce="sum",
        ) # so the output is batch_size x dimPool?
        add_pool = F.elu(add_pool)


        return self.gru(add_pool, graph_embedding)
        





class AttentiveFPReadout(nn.Module): 
    '''
    Compute graph representation from node embeddings using attentive sum readout.
    Code from: https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/readout/attentivefp_readout.py


    Parameters
    ----------
    dimPool : int
        Number of features.
    num_timesteps : int
        Number of succesive readouts.
    dropout : float
    '''
    def __init__(self, dimPool, num_timesteps, dropout):
        super().__init__()

        self.readouts = []
        for _ in range(num_timesteps):
            self.readouts.append(AttentiveFPReadout_step(dimPool, dropout))
        self.readouts = nn.ModuleList(self.readouts)



    def forward(self, x, batch):
        graph_embedding = geom_nn.global_add_pool(x, batch) # initial graph embedding
        for readout in self.readouts:
            graph_embedding = readout(x, graph_embedding, batch)
        return graph_embedding






class GraphCCS(nn.Module):
    '''
    Architecture of GraphCCS (https://doi.org/10.1016/j.chemolab.2024.105177).


    Parameters
    ----------
    node_dim : int
        Number of node-level features.
    edge_dim : int
        Number of edge-level features.
    GCdim : int, default = 400
        Output feature dimension of the graph convolution attention layer and
        of the linear transformation after the graph conlution attention layer.
    n_gcn_layers : int, default = 40
        Number of graph convolution attention layers.
    n_attentive_readouts : int, default = 2
        Output feature dimension is also GCdim.
    dropout : float, default = 0.1
        Applied to the graph convolution attention layers and linear transformation after it.
    '''
    def __init__(
            self,
            node_dim, edge_dim,
            GCdim=400, n_gcn_layers=40,
            n_attentive_readouts = 2,
            dropout=0.1
    ):
        super().__init__()

        self.node_embedder = nn.Linear(node_dim, GCdim) # cast from 58 to GCdim
        self.edge_embedder = nn.Linear(edge_dim, GCdim) # cast from 36 to GCdim
 
        self.gcn_layers = []
        for depth_idx in range(n_gcn_layers):
            self.gcn_layers += [ResAttGCN(GCdim=GCdim, depth_idx=depth_idx, dropout=dropout)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.readout = AttentiveFPReadout(dimPool=GCdim, num_timesteps=n_attentive_readouts, dropout=dropout)

        # CCS feed forward
        self.ccs = nn.Sequential(
            nn.Linear(GCdim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )



    def get_embedding(self, x, edge_index, edge_attr, batch):
        # 1. obtain node-level embeddings
        node_embedding = self.node_embedder(x)
        edge_embedding = self.edge_embedder(edge_attr)

        # 2. perform graph attention
        for gcn_layer in self.gcn_layers:
            node_embedding = gcn_layer(node_embedding, edge_index, edge_embedding)

        # 3. obtain graph-level embedding
        graph_embedding = self.readout(node_embedding, batch)
        return graph_embedding



    def forward(self, x, edge_index, edge_attr, batch):
        graph_embedding = self.get_embedding(x, edge_index, edge_attr, batch)
        return self.ccs(graph_embedding)




        

#######################################################################################
############################### CCSreloaded (heterogeneous) ###########################
#######################################################################################
class HeteroCCS(nn.Module):
    '''
    GNN on heterogeneous 2D SIMGs to predict collision cross-sections (CCSs).
    Fundamentally, it stacks different GNN layers, concatenates the intermediate outputs 
    and uses a feed-forward NN to predict the CCS.


    Parameters
    ----------
    node_types : list
        Names of the node types in the SIMG.
    edge_types : list
        Names of the edge types in the SIMG with specified features.
    featureless_edges : list
        Names of the edge types in the SIMG with unspecified features.
    edge_dims : dict
        Edge-feature width of each featured interaction.
    GATdim : int, default = 256
        Output feature width of each GAT attention head.
    nGATheads : int, default = 4
        Number of GAT heads per layer.
    nGATs : int, default = 3
        Number of GAT layers.
    node_embedding_dim : int, default = 512
        Feature width of the node embeddings after feed-forward.
    dropout : float, default = 0
        Dropout probability in before final projection MLP prediction head.
    '''
    def __init__(self, 
                 node_types, edge_types, featureless_edges, edge_dims,
                 GATdim = 256, nGATheads = 4, nGATs = 3,
                 node_embedding_dim = 512,
                 dropout=0.):
        super().__init__()


        ###### 1. graph attention ######
        self.graph_layers = []
        for _ in range(nGATs):
            # In a layer, each node collects messages from its neighbors and updates its own embedding.
            # In the heterogeneous case this happens per relation with a relation-specific transform,
            # and then each node sums the results across all relations pointing into it.
            layer_convs = {}
            for edge in edge_types + featureless_edges:

                if edge in featureless_edges:
                    # possible to run GATs without specified feats, but then in a feature-less interaction
                    # it would just use the node embeddings
                    # SAGE does exactly that without the whole attention overhead: aggregate neighbors and
                    # combine with the node's own embedding
                    layer_convs[edge] = geom_nn.conv.SAGEConv((-1, -1),
                                                              GATdim, # give same importance to featureless interactions
                                                              aggr = 'mean') # default; could put a learnable aggregator, but
                                                                             # these edges are the least informative

                else:
                    layer_convs[edge] = geom_nn.GATv2Conv(
                        (-1, -1), # -1, -1 to adapt to whatever dims happen at that conv because source and destination are different node types with different widths
                        GATdim, edge_dim = edge_dims[edge],
                        heads = nGATheads, concat = False, # avg
                        add_self_loops = False # breaks for difference src and dst
                    )
                                                               
            self.graph_layers.append(geom_nn.conv.HeteroConv(layer_convs, aggr = 'sum'))
            # sum aggr to let each relation contribute with its full signal and let the next layer decide importance
        self.graph_layers = nn.ModuleList(self.graph_layers)


        ###### 2. node projection ######
        # Each node type lives in incompatible feature spaces.
        # This FFN projects all of them into one shared embedding space to be later pooled
        self.node_embedders = {}
        for node_type in node_types:
            node_embedder = nn.Sequential(
                nn.LazyLinear(node_embedding_dim), # auto-detect feat width
                nn.LayerNorm(node_embedding_dim),
                nn.ReLU(),
                nn.Linear(node_embedding_dim, node_embedding_dim),
                nn.ReLU()
            )
            self.node_embedders[node_type] = node_embedder
        self.node_embedders = nn.ModuleDict(self.node_embedders)


        ###### 3. graph-level embedding ######
        # attention pooling learns which nodes matter, 
        # add-pooling later in forward() is size-extensive, which should correlate with CCS.
        # per node type and concatenate
        self.att_pools = {}
        for node_type in node_types:
            att_pool = geom_nn.GlobalAttention(
                gate_nn = nn.Sequential( # attention scores by mapping node features to shape 1 (for node-level gating) or n_features (for feature-level gating)
                    nn.Linear(node_embedding_dim, node_embedding_dim // 2),
                    nn.ReLU(),
                    nn.Linear(node_embedding_dim // 2, 1)
                ),
                nn = nn.Sequential( # maps node features x of shape in_channels to shape out_channels before combining them with the attention scores
                    nn.Linear(node_embedding_dim, node_embedding_dim),
                    nn.ReLU(),
                    nn.Linear(node_embedding_dim, node_embedding_dim)
                )
            )
            self.att_pools[node_type] = att_pool
        self.att_pools = nn.ModuleDict(self.att_pools)


        ###### prediction head ######
        self.ffn_ccs = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_embedding_dim * len(node_types), 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )



    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        # Jumping knowledge net against oversmoothing: keep raw input and intermediate ones, then concat.
        # the FFN later decides which depth matters more
        # https://doi.org/10.48550/arXiv.1806.03536
        jk = {node_type: [x_node] for node_type, x_node in x_dict.items()}
        h = x_dict
        for graph_layer in self.graph_layers:
            h = graph_layer(h, edge_index_dict, edge_attr_dict)
            h = {node_type: F.relu(node_embeddings) for node_type, node_embeddings in h.items()}
            for node_type, node_embeddings in h.items():
                jk[node_type].append(node_embeddings)
        # jk[node_type] is a list of length (1 + nGATs); concatenating gives width:
        # atom: 60 + nGATs * GATdim , lp: 27 + nGATs * GATdim, bond: 9 + nGATs * GATdim


        h = {node_type: self.node_embedders[node_type](torch.cat(jk[node_type], dim=1)) for node_type in self.node_embedders}


        num_graphs = int(batch_dict['atom'].max()) + 1
        readout = []
        for node_type in self.att_pools.keys():
            readout += [
                self.att_pools[node_type](h[node_type], batch_dict[node_type], size = num_graphs),
                geom_nn.global_add_pool(h[node_type], batch_dict[node_type], size = num_graphs)
            ]


        return self.ffn_ccs(torch.cat(readout, dim=1))






class lightning(pl.LightningModule):
    '''
    This is a wrapper around model HeteroCCS to automate training and
    loading from check points.


    Parameters
    ----------
    architecture_params : dict
        Architecture-specific hyperparameters.
    standardization_stats : tuple
        Mean and standard deviation of training CCSs.
    lr : float, default = 0.001
        Learning rate.
    scheduler_step_size : int, default = 10
        Period of learning rate decay (in epoch number).
    scheduler_gamma : float, default = 0.85
        Multiplicative factor of learning rate decay.
    weight_decay : float, default = 0.0005
        L2 penalty for AdamW. LayerNorm and biases are excluded from the regularization.
    '''
    def __init__(self, architecture_params,
                 standardization_stats,
                 lr = 0.001, scheduler_step_size = 10, scheduler_gamma = 0.85, weight_decay=0.0005):
        super().__init__()

        self.save_hyperparameters() # store all the provided arguments under the self.hparams attribute

        ccs_mean, ccs_std = standardization_stats
        self.register_buffer('ccs_mean', ccs_mean) # for GPU de-standardization
        self.register_buffer('ccs_std', ccs_std)


        self.model = HeteroCCS(**architecture_params)



    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        return self.model(x_dict, edge_index_dict, edge_attr_dict, batch_dict)        


        
    # https://pytorch-lightning.readthedocs.io/en/0.10.0/lightning_module.html
    def shared_step(self, batch, stage):
        raw_ccs_hats = self.forward(
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict,
            batch.batch_dict # trace back the concatenated info to each molecule
        ).squeeze(-1)
        raw_loss = F.mse_loss(raw_ccs_hats, batch.CCS)

        self.log(f"{stage}_loss",
                 raw_loss * self.ccs_std ** 2,
                 prog_bar=True, batch_size=raw_ccs_hats.shape[0], on_step=False, on_epoch = True)

        return raw_loss



    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")



    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")



    def configure_optimizers(self):
        ###### remove some params from weight decay
        # https://mbrenndoerfer.com/writing/adamw-optimizer-decoupled-weight-decay
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or 'norm' in name.lower() or name.endswith('.bias'):
                no_decay.append(p) # LayerNorm weights + biases
            else:
                decay.append(p) # weights
        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': self.hparams.weight_decay},
            {'params': no_decay, 'weight_decay': 0}
        ], lr = self.hparams.lr)
        
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size, # every scheduler_step_size epochs
            gamma=self.hparams.scheduler_gamma # multiply by gamma
        )


        return [optimizer], [scheduler]
    


    def predict_step(self, batch, batch_idx): # call as pl.LightningModule.predict(model, dataLoader) --> list of batch results
        # automatically in eval
        # grad off automatically
        raw_ccs_hats = self.forward(
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict
        ).squeeze(-1)
        ccs_hats = raw_ccs_hats * self.ccs_std + self.ccs_mean
        return ccs_hats.cpu().numpy()