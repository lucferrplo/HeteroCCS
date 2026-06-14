'''
Obtain stereoelectronics-infused molecular graphs (SIMGs)
from a molecular identifier, such as SMILES or InChI strings.


References:

    - Advancing molecular machine learning representations with stereoelectronics-infused molecular graphs, 
    Boiko et al. (https://doi.org/10.1038/s42256-025-01031-9)

    
Code based on:

    - https://github.com/gomesgroup/simg
'''
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors

import torch
import numpy as np
from torch_geometric.data import Data, HeteroData

from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch_geometric.nn.conv import PNAConv
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn import MessagePassing
from torch_geometric import nn as geom_nn
import torch.nn.functional as F

from joblib import Parallel, delayed









#######################################################################################
################################### mol --> conformer #################################
#######################################################################################
def extract_conformer_features(molecule):
    '''
    Extract the atomic, bond and coordinate information for the input to SIMG.


    Parameters
    ----------
    molecule : rdkit.Chem.rdchem Mol object or None


    Returns
    -------
    Extracted graph information : tuple or None
        List of atom symbols, numpy.ndarray of shape(n_atoms, 3) with the x, y and z coordinates as well as
        list of tuples of connectivity representing bond atom, bond atom and bond order.
        If any bond order is found to not be integer, None is returned instead.
        No explicit hydrogens present.
    '''
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]


    conformer = molecule.GetConformer()
    atom_xyz_coordinates = np.asarray(
        [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z] 
         for i in range(molecule.GetNumAtoms())],
        dtype=np.float64
    )


    connectivity = []
    Chem.Kekulize(molecule, clearAromaticFlags=False) # integer bond orders are expected by SIMG code
    for bond in molecule.GetBonds():
        first_atom = bond.GetBeginAtomIdx()
        second_atom = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()
        if not np.isclose(bond_order, round(bond_order)):
            return None
        bond_order = int(round(bond_order))
        connectivity.append((first_atom, second_atom, bond_order))

    

    return atom_symbols, atom_xyz_coordinates, connectivity






#######################################################################################
########################### obtain input molecular graph ##############################
#######################################################################################
class OneHotEncoder():
    '''
    Turn categorical data (with unique categories) into one-hot-encoded vectors (0s and 1s).

    This constructor initializes the class.
    All key attributes are initialized by default (empty dictionary).


    Parameters
    ----------
    None


    Attributes
    ----------
    category_index : dict
        Dictionary that maps the category labels (keys) to its appearance index in the fitting input.
    '''
    def __init__(self):
        self.category_index = {}



    def fit(self, x):
        '''
        Save a dictionary that maps the category labels (keys) to its appearance index in the input.


        Parameters
        ----------
        x : iterable of length n_train


        Returns
        -------
        None
            Updates the following attributes:
                - category_index : dictionary that maps the category labels (keys) to its appearance index in the input.
        '''
        for i, j in enumerate(x):
            self.category_index[j] = i



    def transform(self, x):
        '''
        Transform a category label into a one-hot-encoded vector.


        Parameters
        ----------
        x : category to be one-hot-encoded


        Returns
        -------
        numpy.ndarray of shape (n_train,)
        '''
        Xencoded = np.zeros(len(self.category_index))
        try:
            Xencoded[self.category_index[x]] = 1.
        except KeyError as e:
            raise ValueError(f'Unknown label: {x}.') from e
        return Xencoded






def getLPIMG(atom_symbols, connectivity):
    '''
    Assemble the input molecular graph for the lone pair model.


    Parameters
    ----------
    atom_symbols : list
        List of atom symbol strings.
    connectivity : list
        List of tuples of connectivity representing bond atom, bond atom and bond order.

    
    Returns
    -------
    graph : torch_geometric.data.Data
        Data bidirectional graph with the following attributes:
            - x node feature matrix of shape (n_atoms, 16) for the one-hot-encoded atom symbols
            - edge_index index pairs for each bond of shape (2, n_bonds*2)
            - edge_attr one-hot-encoded bond orders as a matrix of shape (n_bonds*2, 4), 
            where each column contains a 1 for the corresponding
            row bond if it is single, double, triple or quadruple, otherwise 0.
    '''
    # turn into bidirectional graph
    connectivity = connectivity + [(j, i, bond_order) for i, j, bond_order in connectivity]


    # one-hot-encode atom symbols into 16 classes
    atom_symbol21hot = OneHotEncoder()
    atom_symbol21hot.fit(['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi'])
    x = torch.tensor(
        [atom_symbol21hot.transform(atom_symbol) for atom_symbol in atom_symbols],
        dtype=torch.float32
    )


    # obtain the atom idx pairs for each bond
    edge_index = torch.tensor([[x[0], x[1]] for x in connectivity]).t() # 2 x n_bonds*2


    # one-hot-encoded bond orders
    edge_attr = np.zeros((len(connectivity), 4)) # single, double, triple and quadruple bonds
    for i, (_, _, bond_order) in enumerate(connectivity):
        edge_attr[i, bond_order - 1] = 1
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)


    
    graph = Data(
        x=x, # node feature matrix
        edge_index=edge_index, # atom idx pairs
        edge_attr=edge_attr, # one-hot-encoded bond order
    )
    return graph






#######################################################################################
############################## obtain lone pair features ##############################
#######################################################################################
n_atom_features = 16 # size of the atom symbol set
n_lp_classes = 5 # count classes 0, 1, 2, 3, 4
class GNN_LP_Model(nn.Module): # code from https://github.com/gomesgroup/simg


    def __init__(self, hidden_size, deg):
        super().__init__()
        self.layers = []
        
        last_hs = n_atom_features

        self.aggregators = ["sum", "mean", "max", "min", "std"]
        self.scalers = ["identity"]

        for hs in hidden_size:
            self.layers += [
                PNAConv(
                    last_hs, hs, self.aggregators, self.scalers, deg=deg, edge_dim=4
                ),
                nn.ReLU(),
            ]

            last_hs = hs

        self.layers.append(GCNConv(last_hs, n_atom_features))
        self.layers = nn.ModuleList(self.layers)

        self.fcn_head = nn.Sequential(
            nn.Linear(n_atom_features * 2, 10), nn.ReLU(), nn.Linear(10, n_lp_classes * 2)
        )


    def forward(self, x, edge_index, edge_attr):
        out = x

        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                if isinstance(layer, PNAConv):
                    out = layer(out, edge_index, edge_attr)
                else:
                    out = layer(out, edge_index)
            else:
                out = layer(out)

        out = torch.cat((out, x), dim=1)
        out = self.fcn_head(out)

        return out # vector with number of lone pairs per atom and how many are conjugated (p-character), n_atoms x 10






class GNN_LP(LightningModule): # code from https://github.com/gomesgroup/simg


    def __init__(self, config, deg):
        super().__init__()

        hidden_size = config["hidden_size"]
        self.lr = config["lr"]
        self.wd = config["wd"]
        self.batch_size = config["batch_size"]

        self.model = GNN_LP_Model(hidden_size, deg)

        self.aggregators = self.model.aggregators
        self.scalers = self.model.scalers

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        return self.model(x, edge_index, edge_attr)






lp_model = GNN_LP.load_from_checkpoint("lp_pred_model.ckpt")
lp_model.eval()
def getLPs(LPIMG, atom_symbols):
    '''
    Obtain features related to the lone pairs.


    Parameters
    ----------
    LPIMG : torch_geometric.data.Data
        Input molecular graph for the lone pair model with the following attributes:
            - x node feature matrix of shape (n_atoms, 16) for the one-hot-encoded atom symbols
            - edge_index index pairs for each bond of shape (2, n_bonds*2)
            - edge_attr one-hot-encoded bond orders as a matrix of shape (n_bonds*2, 4), 
            where each column contains a 1 for the corresponding
            row bond if it is single, double, triple or quadruple, otherwise 0.
    atom_symbols : list
        List of atom symbol strings.

    
    Returns
    -------
    lp_features : numpy.ndarray of shape (n_lone_pairs, 19) if n_lone_pairs > 0
        The first 16 columns contain the one-hot-encoded atom symbol for each lone pair,
        the 17th if the lone pair is conjugated or not,
        the 18th column contains the total number of lone pairs per corresponding atom and
        the last colum indicates the atom index of the lone pair,
        else numpy.ndarray of shape (0, 19)
    '''
    with torch.no_grad():
        lp_predictions = lp_model(LPIMG)
    n_lps = lp_predictions[:, :5].argmax(dim=1).tolist() # number of lone pairs per atom
    n_conj_lps = lp_predictions[:, 5:].argmax(dim=1).tolist() # number of conjugated lone pairs (p-character)



    if sum(n_lps) > 0: # otherwise there is no point in executing


        # sanity checks
        for i, (n_lp, n_conj_lp, atom_symbol) in enumerate(zip(n_lps, n_conj_lps, atom_symbols)):

            if atom_symbol == "H" and n_lp > 0: # hydrogen cannot have lone pairs in this model
                n_lps[i] = 0
                n_conj_lps[i] = 0

            if n_conj_lp > n_lp: # n_conj_lps cannot be larger than n_lps, so clip if needed
                n_conj_lps[i] = n_lp
        

        # lp features is going to be in the meantime the same atom features but the rows of the atoms repeated
        # depending on the number of lone pairs each has
        lp_extended_atom_list = [] # repeat each atom idx depending on the number of lone pairs
        for i, n_lp in enumerate(n_lps):
            lp_extended_atom_list.extend([i] * n_lp)
        lp_features = LPIMG.x[lp_extended_atom_list].numpy()


        # do the same repetition but not depending on the number of conjugated lone pairs
        conj_lp_extended_atom_list = []
        for i, n_conj_lp in enumerate(n_conj_lps):
            conj_lp_extended_atom_list.extend([i] * n_conj_lp)
        # get a list that indicates if the lone pair is conjugated
        # by default the first lone pairs are assigned to the conjugation
        lp_is_conj = []
        for i in lp_extended_atom_list:
            if i in conj_lp_extended_atom_list:
                lp_is_conj.append(1)
                conj_lp_extended_atom_list.remove(i)
            else:
                lp_is_conj.append(0)

        # reorder the conjugated lone pairs to be the last ones (just to match their original code?)
        lp_atom_idxs = list(set(lp_extended_atom_list)) # atom idxs with conjugated pairs
        new_lp_is_conj = []
        for lp_atom_idx in lp_atom_idxs:
            # get the idxs in lp_is_conj that correspond to the current atom
            idxs = [i for i, x in enumerate(lp_extended_atom_list) if x == lp_atom_idx]
            lp_is_conj_vals = [lp_is_conj[i] for i in idxs]
            # invert list
            new_lp_is_conj.extend(lp_is_conj_vals[::-1])
        lp_is_conj = new_lp_is_conj


        lp_features = np.hstack(
            (
                lp_features,
                np.asarray(lp_is_conj)[:, None],
                np.asarray([lp_extended_atom_list.count(atom) for atom in lp_extended_atom_list])[:, None], # number of lps per atom
                np.asarray(lp_extended_atom_list)[:, None]
            )
        )



    else: # no lone pairs present
        lp_features = np.zeros((0, 19), dtype=np.float32)
    


    return lp_features






#######################################################################################
#################################### obtain SIMG ######################################
#######################################################################################
def getSIMGinput(LPIMG, lp_features, atom_symbols, atom_xyz_coordinates, connectivity):
    '''
    Prepare graph for stereoelectronics calculations.


    Parameters
    ----------
    LPIMG : torch_geometric.data.Data
        Input molecular graph used previously for the lone pair model with the following attributes:
            - x node feature matrix of shape (n_atoms, 16) for the one-hot-encoded atom symbols
            - edge_index index pairs for each bond of shape (2, n_bonds*2)
            - edge_attr one-hot-encoded bond orders as a matrix of shape (n_bonds*2, 4), 
            where each column contains a 1 for the corresponding
            row bond if it is single, double, triple or quadruple, otherwise 0.
    lp_features : numpy.ndarray of shape (sum(lone_pairs), 19) if sum(lone_pairs) > 0
        The first 16 columns contain the one-hot-encoded atom symbol for each lone pair,
        the 17th if the lone pair is conjugated or not,
        the 18th column contains the total number of lone pairs per corresponding atom and
        the last colum indicates the atom index of the lone pair,
        else numpy.ndarray of shape (0, 19)
    atom_symbols : list
        List of atom symbol strings.
    atom_xyz_coordinates : numpy.ndarray of shape(n_atoms, 3) with the x, y and z coordinates for each atom.
    connectivity : list
        List of tuples of connectivity representing bond atom, bond atom and bond order.
    
        
    Returns
    -------
    SIMGinput : torch_geometric.data.Data
        Input molecular graph for stereoelectroncis calculations with the following attributes:
            - x : nodes of shape (n_atoms + n_lps + n_bonds, 16 + 18 + 2 + 3)
            - xyzs : 3D coordinates of shape (n_atoms + n_lps + n_bonds, 3)
            - between_nodes_vectors : 3D distance between nodes of shape (n_atoms + n_lps + n_bonds, 3)
            - atom2bond_idxs : tuple of atom_idx and the bond node idx of shape (2, 2 * n_bonds)
            - edge_index : edge idxs of shape (2, 2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds)
            - orbital_interaction_idxs : edges between orbital-like nodes only: lone pairs and bond orbitals, shape (2, (n_lps + n_bonds) * (n_lps + n_bonds - 1))
            - edge_attr : edge attributes of shape (2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds, 4 + 18 + 2 + 3)
            - symbols : node symbols of shape (n_atoms + n_lps + n_bonds,)
    '''
    n_bonds = sum([x[-1] for x in connectivity])
    n_atoms = atom_xyz_coordinates.shape[0]
    lp_atom_idxs = lp_features[:, -1].astype(int)
    n_lps = lp_atom_idxs.size
    

    connectivity_per_bond = [] # list of bonded atoms per single bond
    one_hot_encoded_bond_order = [] # turn int bond orders into one-hot-encoded bond orders
    bidirectional_connectivity = [] # connected atom idxs duplicated for bidirection graph
    for first_atom_idx, second_atom_idx, bond_order in connectivity:

        for _ in range(bond_order):
            connectivity_per_bond.append([first_atom_idx, second_atom_idx])

        one_hot_encoding = [0, 0, 0, 0]
        one_hot_encoding[bond_order - 1] = 1
        one_hot_encoded_bond_order.append(one_hot_encoding)
        one_hot_encoded_bond_order.append(one_hot_encoding) # duplicate for bidirectional graph

        bidirectional_connectivity.append([first_atom_idx, second_atom_idx])
        bidirectional_connectivity.append([second_atom_idx, first_atom_idx]) # duplicate for bidirectional graph  

    one_hot_encoded_bond_order = np.asarray(one_hot_encoded_bond_order)



    ##################################################
    #################### nodes #######################
    ##################################################


    ###### 3D coordinates ######
    xyzs = atom_xyz_coordinates.tolist()

    # represent lone pairs as extra nodes with the coordinates of the corresponding atoms
    xyzs += atom_xyz_coordinates[lp_atom_idxs].tolist() 
    # each single bond is also represented as extra nodes located halfway between the two bonded atoms
    xyzs += [((np.asarray(xyzs[i]) + np.asarray(xyzs[j])) / 2).tolist() for i, j in connectivity_per_bond]
    xyzs = torch.FloatTensor(xyzs)

    bond_lengths = [
        np.linalg.norm(xyzs[first_atom_idx] - xyzs[second_atom_idx])
        for first_atom_idx, second_atom_idx in connectivity_per_bond
    ]


    ###### π character ######
    # first time a bond appears in connectivity_per_bond is the single bond, the rest account for higher bond orders
    is_pi_bond = [] # 0s and 1s indicated character
    last = None
    for i in connectivity_per_bond:
        if i == last:
            is_pi_bond.append(1)
        else:
            is_pi_bond.append(0)
        last = i


    bond_info = np.vstack((bond_lengths, is_pi_bond)).T
    bond_edges = []
    for bond_edge in bond_info:
        bond_edges.append(bond_edge)
        bond_edges.append(bond_edge)  # duplicate because each bond connects to two atoms
    bond_edges = np.vstack((bond_edges, bond_edges)) # duplicate for bidirectional graph


    # Euclidean vectors between each node in the graph
    between_nodes_vectors = [[0, 0, 0]] * (n_atoms + n_lps) # these are the atoms and the lone pairs, 0 since they are to themselves
    between_nodes_vectors += [ # the bonds
        (np.asarray(xyzs[first_atom_idx]) - np.asarray(xyzs[second_atom_idx])).tolist()
        for first_atom_idx, second_atom_idx in connectivity_per_bond
    ]
    between_nodes_vectors = torch.FloatTensor(between_nodes_vectors)



    ##################################################
    #################### edges #######################
    ##################################################


    ###### connect atom to lone pair and back for bidirectionality ######
    atom_has_lp = []
    lp_relates2atom = []
    for lone_pair_idx, atom_idx in enumerate(lp_features[:, -1]):
        atom_has_lp.append(
            [atom_idx, lone_pair_idx + LPIMG.x.shape[0]] # first come atoms, then lone pair nodes
        )
        lp_relates2atom.append(
            [lone_pair_idx + LPIMG.x.shape[0], atom_idx]
        )


    ###### connect atom nodes to bond nodes obtaining connecting idxs
    atom2bond_idxs = []
    bond_cumsum = 0
    for i, (first_atom_idx, second_atom_idx, bond_order) in enumerate(connectivity):
        for j in range(bond_order):
            if j > 0:
                bond_cumsum += 1
            bond_node_idx = n_atoms + n_lps + i + bond_cumsum # atom, lone pairs and then bond nodes
            atom2bond_idxs.append([first_atom_idx, bond_node_idx])
            atom2bond_idxs.append([second_atom_idx, bond_node_idx])


    ###### building edges between orbital-like nodes only: lone pairs and bond orbitals
    orbital_connections = [
        (i, j)
        for i in range(n_atoms, n_atoms + n_lps + n_bonds)
        for j in range(n_atoms, n_atoms + n_lps + n_bonds)
        if i != j
    ]



    ##################################################
    ########### torch_geometric.data.Data ############
    ##################################################
    graph = Data()


    ###### atom, lone pairs and bond nodes ######
    graph.x = torch.block_diag(
        LPIMG.x,
        torch.FloatTensor(lp_features[:, :-1]),
        torch.FloatTensor(bond_info)
    )

    is_atom = [1] * n_atoms + [0] * n_lps + [0] * n_bonds
    is_lp = [0] * n_atoms + [1] * n_lps + [0] * n_bonds
    is_bond = [0] * n_atoms + [0] * n_lps + [1] * n_bonds
    graph.x = torch.cat(
        (graph.x, torch.FloatTensor([is_atom, is_lp, is_bond]).T), dim=1
    )    


    ###### vectorial data ######
    graph.xyzs = xyzs
    graph.between_nodes_vectors = between_nodes_vectors


    ###### indexes
    graph.atom2bond_idxs = torch.LongTensor(atom2bond_idxs).T

    atom2bond_idxs = atom2bond_idxs + [[j, i] for i, j in atom2bond_idxs] # bidirectional graph
    # add these edges to the existing ones
    graph.edge_index = bidirectional_connectivity + atom_has_lp + lp_relates2atom + atom2bond_idxs # take all lists from edges, flatten and append: n_edges, 2
    graph.edge_index = [[int(num) for num in item] for item in list(graph.edge_index)] # make sure ints for indexing
    graph.edge_index = torch.LongTensor(graph.edge_index).T

    graph.orbital_interaction_idxs = torch.LongTensor(orbital_connections).T


    ###### edge features ######
    lps = np.vstack((lp_features[:, :-1], lp_features[:, :-1])) # duplicate for bidirectional graph
    graph.edge_attr = torch.block_diag(
        torch.FloatTensor(one_hot_encoded_bond_order),
        torch.FloatTensor(lps), 
        torch.FloatTensor(bond_edges)
    )
    is_atom_atom = np.asarray([1] * one_hot_encoded_bond_order.shape[0] + [0] * lps.shape[0] + [0] * bond_edges.shape[0])
    is_atom_lp = np.asarray([0] * one_hot_encoded_bond_order.shape[0] + [1] * lps.shape[0] + [0] * bond_edges.shape[0])
    is_orbital = np.asarray([0] * one_hot_encoded_bond_order.shape[0] + [0] * lps.shape[0] + [1] * bond_edges.shape[0])
    graph.edge_attr = torch.column_stack((graph.edge_attr, torch.FloatTensor(is_atom_atom), torch.FloatTensor(is_atom_lp), torch.FloatTensor(is_orbital)))


    ###### graph node symbols ######
    graph.symbols = atom_symbols + ["LP"] * n_lps + ["BND"] * n_bonds
    # graph.is_atom = torch.Tensor(is_atom)
    # graph.is_lp = torch.Tensor(is_lp)
    # graph.is_bond = torch.Tensor(is_bond)



    return graph






class MLP(nn.Module): # code from https://github.com/gomesgroup/simg
    def __init__(self, input_size, layers, activation='ReLU'):
        super().__init__()

        self.mlp = []
        current_hidden_size = input_size

        for layer in layers:
            self.mlp.append(nn.Linear(current_hidden_size, layer))

            if activation == 'ReLU':
                self.mlp.append(nn.ReLU())
            elif activation == 'Sigmoid':
                self.mlp.append(nn.Sigmoid())
            elif activation == 'Tanh':
                self.mlp.append(nn.Tanh())
            else:
                raise ValueError('Unknown activation')

            current_hidden_size = layer

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)






class NodeEvolver(nn.Module): # code from https://github.com/gomesgroup/simg
    def __init__(self, embedding_dim, hidden_size, node_target_size, hidden_layers=(256, 128), target_layers=(256, 128),
                 out_transform=()):
        super().__init__()

        self.hidden_transform = MLP(hidden_size, hidden_layers)
        self.target_transform = MLP(node_target_size + embedding_dim, target_layers)

        self.out_transform = MLP(hidden_layers[-1] * 3, list(out_transform) + [hidden_size], 'Tanh')

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding, hidden, node_target, mask):
        hidden_transformed = self.hidden_transform(hidden)  # [N_h, H]
        target_transformed = self.target_transform(torch.hstack((node_target, embedding)))  # [N_t, H]

        weights = self.softmax(
            (hidden_transformed @ target_transformed.T) * mask
        )  # [N_h, N_t] 
        weighted = weights @ target_transformed  # [N_h, H]

        out = hidden + self.out_transform(torch.hstack((weighted, hidden_transformed, target_transformed)))

        return out






class GNNModel(nn.Module): # code from https://github.com/gomesgroup/simg
    def __init__(self, hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim, gnn_output_dim, heads,
                 use_gnn=True, take_last_only=False, baseline_gnn=None, hidden_dim=256,
                 use_evolver=False, evolver_steps=5):
        super().__init__()
        self.layers = []

        n_atom_features = 17
        n_lp_features = 19
        n_bond_features = 3
        n_edge_features = 27

        n_atom_targets = 4
        n_lp_targets = 5
        n_bond_targets = 7
        n_int_targets = 3

        last_hs = n_atom_features + n_lp_features + n_bond_features

        self.use_gnn = use_gnn
        self.take_last_only = take_last_only
        self.baseline_gnn = baseline_gnn

        if use_gnn:
            if baseline_gnn is None:
                print('Using full GNN model')

                all_hss = 0

                for hs in hidden_size:
                    self.layers += [
                        geom_nn.GATConv(last_hs, hs, edge_dim=n_edge_features, heads=heads, concat=False),
                        nn.ReLU(),
                    ]

                    last_hs = hs
                    all_hss += hs

                self.layers.append(geom_nn.GCNConv(last_hs, gnn_output_dim))
                all_hss += gnn_output_dim

                self.layers = nn.ModuleList(self.layers)
            else:
                print('Using baseline GNN model:', baseline_gnn)

                model_dict = {
                    'GAT_tg': geom_nn.GAT,
                    'GCN_tg': geom_nn.GraphSAGE,
                }

                model_params = {
                    'in_channels': n_atom_features + n_lp_features + n_bond_features,
                    'hidden_channels': 1024,
                    'out_channels': gnn_output_dim,
                    'num_layers': 7
                }

                if baseline_gnn == 'GAT_tg':
                    model_params['edge_dim'] = n_edge_features

                self.model = model_dict[baseline_gnn](**model_params)

            if take_last_only:
                print('Using only last GNN layer')
                fcn_input_dim = gnn_output_dim
            else:
                print('Stacking all GNN layer outputs')
                fcn_input_dim = n_atom_features + n_lp_features + n_bond_features + all_hss
        else:
            fcn_input_dim = n_atom_features + n_lp_features + n_bond_features

        self.fcn_head = nn.Sequential(
            nn.Linear(
                fcn_input_dim, fcn_hidden_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(fcn_hidden_dim),
            nn.Linear(fcn_hidden_dim, embedding_dim),
            nn.ReLU()
        )

        if use_evolver:
            print('Using evolver')
            embedding_dim += hidden_dim

        self.link_clf = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + 1, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, 1)
        )

        self.fcn_head_a2b = nn.Sequential(
            nn.Linear(embedding_dim * 2, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, 6)
        )

        self.fcn_head_node = nn.Sequential(
            nn.Linear(embedding_dim, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, n_atom_targets + n_lp_targets + n_bond_targets)
        )

        self.fcn_int_node = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + 1, clf_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(clf_hidden_dim),
            nn.Linear(clf_hidden_dim, n_int_targets)
        )

        self.hidden_dim = hidden_dim
        self.evolver = NodeEvolver(embedding_dim, hidden_dim,
                                   n_atom_targets + n_lp_targets + n_bond_targets)

        self.use_evolver = use_evolver
        self.evolver_steps = evolver_steps

    def get_predictions(self, embeddings, interaction_edge_index_pos, interaction_edge_index, xyz_data, vector_data,
                        a2b_index):

        # Link prediction task
        link_preds = embeddings[interaction_edge_index]
        link_preds = torch.cat([link_preds[0], link_preds[1]], axis=1)

        positions = xyz_data[interaction_edge_index]
        vectors = vector_data[interaction_edge_index]

        distances = ((positions[0] - positions[1]) ** 2).sum(axis=1)
        angles = F.cosine_similarity(vectors[0], vectors[1])

        link_preds = torch.hstack([link_preds, distances[:, None], angles[:, None]])
        link_preds = self.link_clf(link_preds)

        # A2b prediction
        a2b_preds = embeddings[a2b_index]
        a2b_preds = torch.cat([a2b_preds[0], a2b_preds[1]], axis=1)
        a2b_preds = self.fcn_head_a2b(a2b_preds)

        # Node prediction
        node_preds = self.fcn_head_node(embeddings)

        # Interaction energy prediction
        int_preds = embeddings[interaction_edge_index_pos]
        int_preds = torch.cat([int_preds[0], int_preds[1]], axis=1)

        positions = xyz_data[interaction_edge_index_pos]
        vectors = vector_data[interaction_edge_index_pos]

        distances = ((positions[0] - positions[1]) ** 2).sum(axis=1)
        angles = F.cosine_similarity(vectors[0], vectors[1])

        int_preds = torch.hstack([int_preds, distances[:, None], angles[:, None]])
        int_preds = self.fcn_int_node(int_preds)

        return link_preds, a2b_preds, node_preds, int_preds

    def forward(self, x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                vector_data, a2b_index, mask):
        if (self.evolver_steps < 1) and self.use_evolver:
            raise ValueError('evolver_steps must be at least 1')

        embeddings = self.get_embedding(x, edge_index, edge_attr)

        intermediate = []
        intermediate_pred = []

        if self.use_evolver:
            hidden_embedding = torch.rand(embeddings.shape[0], self.hidden_dim).to('cuda:0')
            intermediate.append(hidden_embedding.detach().cpu().numpy())

            for evolver_step in range(self.evolver_steps):
                prediction_embedding = torch.hstack([embeddings, hidden_embedding])
                link_preds, a2b_preds, node_preds, int_preds = self.get_predictions(prediction_embedding,
                                                                                    interaction_edge_index_pos,
                                                                                    interaction_edge_index, xyz_data,
                                                                                    vector_data, a2b_index)

                intermediate_pred.append(node_preds.detach().cpu().numpy())

                if evolver_step != self.evolver_steps - 1:  # If not last step
                    hidden_embedding = self.evolver(embeddings, hidden_embedding, node_preds, mask)
                    intermediate.append(hidden_embedding.detach().cpu().numpy())

        else:
            link_preds, a2b_preds, node_preds, int_preds = self.get_predictions(embeddings,
                                                                                interaction_edge_index_pos,
                                                                                interaction_edge_index, xyz_data,
                                                                                vector_data, a2b_index)

        return link_preds, a2b_preds, node_preds, int_preds, intermediate, intermediate_pred

    def get_embedding(self, x, edge_index, edge_attr):
        out = x

        if self.use_gnn:
            if self.baseline_gnn is None:
                each_step = [x]

                for layer in self.layers:
                    if isinstance(layer, geom_nn.MessagePassing):
                        if isinstance(layer, geom_nn.GATConv):
                            out = layer(out, edge_index, edge_attr)
                        else:
                            out = layer(out, edge_index)
                    else:
                        out = layer(out)
                        each_step.append(out)

                each_step.append(out)

                if self.take_last_only:
                    out = each_step[-1]
                else:
                    out = torch.cat(each_step, dim=1)
            else:
                if self.baseline_gnn in ['GCN_tg']:
                    out = self.model(x, edge_index)
                elif self.baseline_gnn == 'GAT_tg':
                    out = self.model(x, edge_index, edge_attr)
                else:
                    raise ValueError('Unknown model type')

        out = self.fcn_head(out)

        return out






class GNN(pl.LightningModule): # code from https://github.com/gomesgroup/simg
    def __init__(self,
                 hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim, gnn_output_dim, heads,
                 lr=2e-4, use_gnn=True, take_last_only=False, baseline_gnn=None,
                 hidden_dim=256, use_evolver=False, evolver_steps=5, perform_matching=False):
        super().__init__()
        self.save_hyperparameters()

        self.model = GNNModel(hidden_size, fcn_hidden_dim, clf_hidden_dim, embedding_dim,
                              gnn_output_dim, heads, use_gnn, take_last_only, baseline_gnn,
                              hidden_dim, use_evolver, evolver_steps
                              )

    def forward(self, x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                vector_data, a2b_index, mask=None):
        if self.hparams.use_evolver:
            assert mask is not None, 'Mask should not be None if use_evolver is True'
        return self.model(x, edge_index, edge_attr, interaction_edge_index_pos, interaction_edge_index, xyz_data,
                          vector_data, a2b_index, mask)






def atom2features(atom, contribs, asa, tpsa, HAcceptors, HDonors,
                  atom_hybridization_encoder, atom_nHs_encoder, atom_degree_encoder, atom_ringsize_encoder):
    '''
    Calculates a set of 39 features for an atom in a molecule.


    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
    contribs : list of length n_atoms
        Each element is a tuple of 2 values: the first one is the atom contribution to the logP
        and the second one to the molar refractivity.
    asa : list of length n_atoms
        Atoms's contribution to Labute's approximation to total solvent-accesible surface area.
    tpsa : list of length n_atoms
        Atom's contribution to total topological polar surface area.
    HAcceptors : list of length n_HAcceptors
        Indexes of atoms considered H acceptors.
    HDonors : list of length n_HDonors
        Indexes of atoms considered H donors.
    atom_hybridization_encoder : Python class
        One-hot encoder for hybridization types SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED and S.
    atom_nHs_encoder : Python class
        One-hot encoder for number of hydrogen atoms 0, 1, 2, 3 and 4.
    atom_degree_encoder : Python class
        One-hot encoder for number of graph atom degree 0, 1, 2, 3, 4, 5 and 6.
    atom_ringsize_encoder : Python class
        One-hot encoder for maximum ringsize containing the given atom 0, 3, 4, 5, 6, 7, 8, 9 and 10.

    
    Returns
    -------
    numpy.ndarray of shape(39,)
    '''
    ###### bool
    atom_isinring = atom.IsInRing()
    atom_isaromatic = atom.GetIsAromatic()
    atom_ishacceptor = atom.GetIdx() in HAcceptors
    atom_ishdonor = atom.GetIdx() in HDonors
    atom_ischiral = atom.HasProp("_ChiralityPossible")

    bool_feats = [atom_isinring, atom_isaromatic, atom_ishacceptor, atom_ishdonor, atom_ischiral]


    ###### one-hot encoded
    atom_hybridization = atom_hybridization_encoder.transform(atom.GetHybridization())
    atom_nHs = atom_nHs_encoder.transform(min(atom.GetTotalNumHs(),4))
    atom_degree = atom_degree_encoder.transform(min(atom.GetDegree(),6))

    for atom_ringsize in [10,9,8,7,6,5,4,3,0]:
        if atom.IsInRingSize(atom_ringsize):
            break
    atom_ringsize = atom_ringsize_encoder.transform(atom_ringsize)


    ###### continuous
    atom_index = atom.GetIdx()

    GasteigerCharge = atom.GetProp('_GasteigerCharge')
    if GasteigerCharge in ['-nan', 'nan', '-inf', 'inf']:
       GasteigerCharge = 0
    GasteigerCharge = float(GasteigerCharge)
    
    CrippenLogP = contribs[atom_index][0]
    MolarRefrac = contribs[atom_index][1]

    atom_mass = atom.GetMass()

    atom_asa = asa[atom_index] 

    atom_tpsa = tpsa[atom_index]



    atom_features = np.concatenate(
        (
            bool_feats,
            atom_hybridization, atom_nHs, atom_degree, atom_ringsize,
            [GasteigerCharge, CrippenLogP, MolarRefrac, atom_mass, atom_asa, atom_tpsa]
        )
    )
    return atom_features






gnn = GNN.load_from_checkpoint('nbo_pred_model.ckpt')
gnn.eval()
def getSIMG(SIMGinput, threshold = 0.5,
            homogeneous = True, mol = None,
            return_stereoelectronics = False):
    '''
    Build SIMG.


    Parameters
    ----------
    SIMGinput : torch_geometric.data.Data
        Input molecular graph for stereoelectroncis calculations with the following attributes:
            - x : nodes of shape (n_atoms + n_lps + n_bonds, 16 + 18 + 2 + 3)
            - xyzs : 3D coordinates of shape (n_atoms + n_lps + n_bonds, 3)
            - between_nodes_vectors : 3D distance between nodes of shape (n_atoms + n_lps + n_bonds, 3)
            - atom2bond_idxs : tuple of atom_idx and the bond node idx of shape (2, 2 * n_bonds)
            - edge_index : edge idxs of shape (2, 2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds)
            - orbital_interaction_idxs : edges between orbital-like nodes only: lone pairs and bond orbitals, shape (2, (n_lps + n_bonds) * (n_lps + n_bonds - 1))
            - edge_attr : edge attributes of shape (2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds, 4 + 18 + 2 + 3)
            - symbols : node symbols of shape (n_atoms + n_lps + n_bonds,)
    threshold : float
        Sigmoidal threshold to consider an existing interaction.
    homogeneous : bool, default = True
        Whether to build a heterogeneous or homogenous SIMG.
    mol : rdkit.Chem.rdchem.Mol, default = None
        Only needed if homogenous = False.
    return_stereoelectronics : bool, default = False
        Whether to return the predicted stereoelectronics information.


    Returns
    -------
    SIMG : torch_geometric.data.Data if homogeneous == True else torch_geometric.data.HeteroData
        if homogeneous:
            - x : nodes of shape (n_atoms + n_lps + n_bonds + n_pos_interactions, 16 + 18 + 2 + 3 + 16 + 3)
            - edge_index : edge idxs of shape (2, 2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds + 4 * n_bonds + 2 * n_pos_interactions)
            - edge_attr : edge attributes of shape (2 * n_atom_atom_connections + 2 * n_lps + 4 * n_bonds + 4 * n_bonds + 2 * n_pos_interactions, 4 + 18 + 2 + 3 + 6 + 3)
        else:
            - nodes
                - atom : atom nodes of shape (n_atoms, 16 + 39 + 4 + 1)
                - lp : lone pair nodes of shape (n_lps, 17 + 5 + 5)
                - bond : bond orbitals of shape (n_bonds, 2 + 7)
            - edges
                - 'atom', 'bonded_to', 'atom' : standard molecular graph edges of shape (2 * n_sigma_bonds, 12)
                - 'atom', 'has', 'lp' : shape (n_lps, 18)
                - 'lp', 'relates_to', 'atom' : shape (n_lps, 18)
                - 'atom', 'with', 'bond' : shape (2 * n_bonds, 2 + 6)
                - 'bond', 'with', 'atom' : shape (2 * n_bonds, 2)
                - 'lp', 'interacts_with', 'lp' : shape (n_lp2lp, 3)
                - 'lp', 'rev_interacts_with', 'lp' : shape (n_lp2lp, None)
                - 'bond', 'interacts_with', 'bond' : shape (n_bond2bond, 3)
                - 'bond', 'rev_interacts_with', 'bond' : shape (n_bond2bond, None)
                - 'lp', 'interacts_with', 'bond' : shape (n_lp2bond, 3)
                - 'lp', 'rev_interacts_with', 'bond' : shape (n_lp2bond, None)
                - 'bond', 'interacts_with', 'lp' : shape (n_lp2bond, 3)
                - 'bond', 'rev_interacts_with', 'lp' : shape (n_lp2bond, None)
    predicted_stereoelectronics : tuple of length 4, if return_stereoelectronics == True
        - node_node_interaction_probs : numpy.ndarray of shape(n_lps + n_bonds, n_lps + n_bonds)
        Does lone pair/bond orbital i interact with orbital j?
        - atom2bond_preds : torch.Tensor of shape (2 * n_bonds, 6)
        What is the atom-specific contribution to the bond orbital? It is comprised of 6 features,
        which are s, p, d, f characters (raw logits), polarization and polarization coefficient.
        - node_preds : 
        NBO properties of each node: torch.Tensor of shape (n_atoms + n_lps + n_bonds, 16)
            - atom nodes: charge, core electrons, valence electrons and total electrons (4)
            - lone pair nodes: s, p, d, f character (raw logits) and occupancy (5)
            - bond/orbital nodes: s, p, d, f character (raw logits), occupancy, polarizability difference and polarizability difference coefficient polarization (7)
        - interaction_preds : torch.Tensor of shape (pos((n_lps + n_bonds) * (n_lps + n_bonds - 1)), 3)
        If lone pair/bond orbital i interacts with orbital j, what are the numerical NBO donnor --> acceptor interaction values?
        These are 3 NBO features: perturbation energy, energy difference and Fock matrix element.
    '''
    ###### predict interactions ######
    # obtain idxs of all pairwise node interactions
    interaction_edge_index = [[i, j] for i in range(SIMGinput.x.shape[0]) for j in range(SIMGinput.x.shape[0])]
    interaction_edge_index = torch.LongTensor(interaction_edge_index).T

    # inference
    with torch.no_grad():
        node_node_interaction_probs, atom2bond_preds, node_preds, interaction_preds, _, _ = gnn.forward(
            SIMGinput.x, SIMGinput.edge_index, SIMGinput.edge_attr, SIMGinput.orbital_interaction_idxs, interaction_edge_index,
            SIMGinput.xyzs, SIMGinput.between_nodes_vectors, SIMGinput.atom2bond_idxs
        )


    ###### node_node_interaction_probs: does orbital i interact with orbital j?
    # connections between donor and acceptor orbitals (lone pairs or bonds)
    node_node_interaction_probs = torch.sigmoid(
        node_node_interaction_probs.reshape((SIMGinput.x.shape[0], SIMGinput.x.shape[0]))
    ).numpy()
    # node_node_interaction_probs is a square matrix with values between 0 and 1
    # indicating the prbability of presence of an interaction between each corresponding node

    # remove atom-atom interaction to concentrate on donor-acceptor orbitals
    n_atoms = sum([symbol != "LP" and symbol !="BND" for symbol in SIMGinput.symbols])
    node_node_interaction_probs = node_node_interaction_probs[n_atoms:, n_atoms:]

    np.fill_diagonal(node_node_interaction_probs, 0) # remove self-interactions
    
    node_node_interaction_probs[node_node_interaction_probs < threshold] = 0
    node_node_interaction_probs[node_node_interaction_probs > 0] = 1


    ###### atom2bond_preds: what is the atom-specific contribution to the bond orbital?
    # s, p, d, f characters, polarization and polarization coefficient (6)
    # atom2bond_preds = atom2bond_preds.cpu()


    ###### node_preds: what are the NBO properties of each node?
    # atom nodes: charge, core electrons, valence electrons, total electrons (4)
    # lone pair nodes: s, p, d, f character (raw logits, not prob) and occupancy (5)
    # bond/orbital nodes: s, p, d, f character (regressed for MSELoss... not raw logits?), occupancy, polarizability difference and polarizability difference coefficient polarization (7)
    # node_preds = node_preds.cpu()


    ###### interaction_preds: if orbital i interacts with orbital j, what are the numerical NBO interaction values?
    # perturbation energy, energy difference, Fock matrix element (3)
    row_idx, col_idx = np.where(node_node_interaction_probs == 1) # finds the position of all predicted interactions
    pos_orbital_interaction_idxs = np.concatenate((row_idx[:, None], col_idx[:, None]), axis=1) + n_atoms # n_interactions, 2

    all_orbital_interaction_idxs = SIMGinput.orbital_interaction_idxs.T.numpy()
    # at which position in all_orbital_interaction_idxs do I find the predicted positive edges?
    if np.shape(pos_orbital_interaction_idxs)[0] > 1000: # divide orbital interactions into chunks of size 500
        chunks = np.array_split(pos_orbital_interaction_idxs, np.shape(pos_orbital_interaction_idxs)[0] // 500, axis=0)
        def get_int_idx(chunk):
            return np.where((all_orbital_interaction_idxs[:, None] == chunk).all(axis=2))[0]
        original_interaction_idxs = Parallel(n_jobs=-1, verbose = 0)(delayed(get_int_idx)(chunk) for chunk in chunks)
        original_interaction_idxs = np.concatenate(original_interaction_idxs)
    else:
        original_interaction_idxs = np.where((all_orbital_interaction_idxs[:, None] == pos_orbital_interaction_idxs).all(axis=2))[0]
    
    if pos_orbital_interaction_idxs.shape[0] == 0:
        interaction_preds = torch.empty((0, 3), dtype=torch.float32)
        pos_orbital_interaction_idxs = torch.empty((2, 0), dtype=torch.long)
    else:
        interaction_preds = interaction_preds[original_interaction_idxs]
        pos_orbital_interaction_idxs = torch.LongTensor(pos_orbital_interaction_idxs).T



    

    ##################################################
    ############# build homogeneous SIMG #############
    ##################################################
    if homogeneous:
        SIMG = Data()

        SIMG.x = torch.hstack((SIMGinput.x, node_preds))
        SIMG.x = torch.block_diag(SIMG.x, interaction_preds)

        SIMG.edge_index = torch.hstack((
            SIMGinput.edge_index, SIMGinput.atom2bond_idxs,
            torch.LongTensor(SIMGinput.atom2bond_idxs.numpy()[::-1].copy()) # make bidirectional
        ))
        SIMG.edge_index = torch.hstack((
            SIMG.edge_index, pos_orbital_interaction_idxs,
            torch.LongTensor(pos_orbital_interaction_idxs.numpy()[::-1].copy()) # make bidirectional
        ))

        SIMG.edge_attr = torch.block_diag(SIMGinput.edge_attr, torch.vstack([atom2bond_preds] * 2)) # bidirectional
        SIMG.edge_attr = torch.block_diag(SIMG.edge_attr, torch.vstack([interaction_preds] * 2)) # bidirectional

        # SIMG.is_atom = SIMGinput.is_atom
        # SIMG.is_lp = SIMGinput.is_lp
        # SIMG.is_bond = SIMGinput.is_bond

        SIMG.symbols = SIMGinput.symbols
    




    ##################################################
    ############ build heterogeneous SIMG ############
    ##################################################
    else: # heterogeneous molecular graph
        if mol is None:
            raise ValueError("rdkit.Chem.rdchem.Mol must be provided for heterogenous graphs (homogeneous==False).")
        

        SIMG = HeteroData()

        n_lps = sum([symbol == "LP" for symbol in SIMGinput.symbols])
        n_bonds = sum([symbol == "BND" for symbol in SIMGinput.symbols])

        lp_idx_offset = n_atoms
        bond_idx_offset = n_atoms + n_lps



        ###### nodes ######


        ###### atoms: based on one-hot-encoded atom, RDKit features and NBO properties
        # calculate RDKit features
        AllChem.ComputeGasteigerCharges(mol)
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        asa = rdMolDescriptors._CalcLabuteASAContribs(mol)[0] 
        tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
        HAcceptors = [i[0] for i in Lipinski._HAcceptors(mol)]
        HDonors = [i[0] for i in Lipinski._HDonors(mol)]
        atom_hybridization_encoder = OneHotEncoder()
        atom_hybridization_encoder.fit([
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S
        ])
        atom_nHs_encoder = OneHotEncoder()
        atom_nHs_encoder.fit([0, 1, 2, 3, 4])
        atom_degree_encoder = OneHotEncoder()
        atom_degree_encoder.fit([0,1,2,3,4,5,6])
        atom_ringsize_encoder=OneHotEncoder()
        atom_ringsize_encoder.fit([0,3,4,5,6,7,8,9,10])

        rdkit_feats = torch.as_tensor(
            [atom2features(atom, contribs, asa, tpsa, HAcceptors, HDonors,
                           atom_hybridization_encoder, atom_nHs_encoder, atom_degree_encoder, atom_ringsize_encoder)
             for atom in mol.GetAtoms()]
        )

        
        radial_dists = (SIMGinput.xyzs[:n_atoms] - SIMGinput.xyzs[:n_atoms].mean(0, keepdim=True)).norm(dim=1, keepdim=True)

        SIMG['atom'].x = torch.hstack((
            SIMGinput.x[:n_atoms, :16], 
            rdkit_feats.float(), # float63 to float32
            node_preds[:n_atoms, :4], # charge, core electrons, valence electrons, total electrons (4)
            radial_dists # distance to molecule's centroid (rotation invariant, otherwise 
            # same mol with different orientations get different CCSs, which is rotation-averaged)
        ))


        ###### lone pairs
        lp_count_encoder = OneHotEncoder()
        lp_count_encoder.fit([0, 1, 2, 3, 4])
        lp_counts = [lp_count_encoder.transform(int(c)) for c in SIMGinput.x[lp_idx_offset:bond_idx_offset, 33]]
        lp_counts = torch.as_tensor(lp_counts, dtype = torch.float32).reshape(-1, 5) # ensure matching dims in case no lps
        SIMG['lp'].x = torch.hstack((
            SIMGinput.x[lp_idx_offset:bond_idx_offset, 16:33], # lp features
            lp_counts,
            node_preds[lp_idx_offset:bond_idx_offset, 4:(4 + 5)] # s, p, d, f character (raw logits, not prob) and occupancy (5)
        ))
        

        ###### bonds
        SIMG['bond'].x = torch.hstack((
            SIMGinput.x[bond_idx_offset:, (16+18):36], 
            node_preds[bond_idx_offset:, (4 + 5):16] # s, p, d, f character (raw logits, not prob), occupancy, polarizability difference and polarizability difference coefficient polarization (7)
        ))



        ###### edges ######
        def local_edge_index(edge_index, src_offset=0, dst_offset=0): # each interaction requires a local index
            edge_index = edge_index.clone()
            if edge_index.shape[1] > 0:
                edge_index[0] -= src_offset
                edge_index[1] -= dst_offset
            return edge_index
        
        n_edges = SIMGinput.edge_index.shape[1]
        n_atom2atom_edges = n_edges - 2 * n_lps - 4 * n_bonds
        n_atom2lp_edges = n_lps
        n_lp2atom_edges = n_lps
        n_atom2bond_edges = 2 * n_bonds
        n_bond2atom_edges = 2 * n_bonds

        atom2atom_idxs = slice(0, n_atom2atom_edges)
        atom2lp_idxs = slice(atom2atom_idxs.stop, atom2atom_idxs.stop + n_atom2lp_edges)
        lp2atom_idxs = slice(atom2lp_idxs.stop, atom2lp_idxs.stop + n_lp2atom_edges)
        atom2bond_idxs = slice(lp2atom_idxs.stop, lp2atom_idxs.stop + n_atom2bond_edges)
        bond2atom_idxs = slice(atom2bond_idxs.stop, atom2bond_idxs.stop + n_bond2atom_edges)


        SIMG['atom', 'bonded_to', 'atom'].edge_index = local_edge_index(
            # contains forward and backward edges
            # bonds are symmetric --> parameter sharing
            SIMGinput.edge_index[:, atom2atom_idxs],
            src_offset=0,
            dst_offset=0
        )

        SIMG['atom', 'bonded_to', 'atom'].edge_attr = SIMGinput.edge_attr[atom2atom_idxs, :4] # one-hot-encoded bond order
        # some additional bond information
        one_hot_stereo = OneHotEncoder()
        one_hot_stereo.fit([
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ])
        is_conjugated = []
        is_in_ring = []
        bond_stereochemistry = []
        for first_atom_idx, second_atom_idx in zip(
            SIMGinput.edge_index[:, atom2atom_idxs][0],
            SIMGinput.edge_index[:, atom2atom_idxs][1]
        ):
            bond = mol.GetBondBetweenAtoms(int(first_atom_idx), int(second_atom_idx))
            is_conjugated.append(int(bond.GetIsConjugated()))
            is_in_ring.append(bond.IsInRing())
            bond_stereochemistry.append(one_hot_stereo.transform(bond.GetStereo()))
        SIMG['atom', 'bonded_to', 'atom'].edge_attr = torch.hstack((
            SIMG['atom', 'bonded_to', 'atom'].edge_attr,
            torch.as_tensor(is_conjugated, dtype=torch.float)[:, None],
            torch.as_tensor(is_in_ring, dtype=torch.float)[:, None],
            torch.as_tensor(bond_stereochemistry, dtype=torch.float)
        ))


        SIMG['atom', 'has', 'lp'].edge_index = local_edge_index(
            SIMGinput.edge_index[:, atom2lp_idxs],
            src_offset=0,
            dst_offset=lp_idx_offset
        )

        # SIMG['atom', 'has', 'lp'].edge_attr = SIMGinput.edge_attr[atom2lp_idxs, 4:22]
        # use node embeddings
        

        SIMG['lp', 'relates_to', 'atom'].edge_index = local_edge_index( # in separate dict due to different node types
            # reverse for bidirectionality
            # direction has meaning, so avoid shared parameters
            SIMGinput.edge_index[:, lp2atom_idxs],
            src_offset=lp_idx_offset,
            dst_offset=0
        )

        # SIMG['lp', 'relates_to', 'atom'].edge_attr = SIMGinput.edge_attr[lp2atom_idxs, 4:22]
        # use node embeddings


        SIMG['atom', 'with', 'bond'].edge_index = local_edge_index(
            SIMGinput.edge_index[:, atom2bond_idxs],
            src_offset=0,
            dst_offset=bond_idx_offset
        )

        SIMG['atom', 'with', 'bond'].edge_attr = torch.hstack((
            SIMGinput.edge_attr[atom2bond_idxs, 22:24].float(), # bond length + π character
            atom2bond_preds.float() # atom-specific bond contribution
        ))


        SIMG['bond', 'with', 'atom'].edge_index = local_edge_index(
            SIMGinput.edge_index[:, bond2atom_idxs],
            src_offset=bond_idx_offset,
            dst_offset=0
        )

        SIMG['bond', 'with', 'atom'].edge_attr = SIMGinput.edge_attr[bond2atom_idxs, 22:24].float() # bond length + π character
        # atom2bond_preds.float()
        # atom2bond_preds is intrinsically a directional edge
        # so it is not chemically faithful to add the reverse message-passing edge with
        # interaction features. Therefore, I just append the bond-generic features.



        ###### orbital interactions (donor --> acceptor) ######
        src = pos_orbital_interaction_idxs[0] # donor
        dst = pos_orbital_interaction_idxs[1] # acceptor
        src_is_lp = (src >= lp_idx_offset) & (src < bond_idx_offset)
        dst_is_lp = (dst >= lp_idx_offset) & (dst < bond_idx_offset)
        src_is_bond = src >= bond_idx_offset
        dst_is_bond = dst >= bond_idx_offset
        lp2lp_mask = src_is_lp & dst_is_lp
        lp2bond_mask = src_is_lp & dst_is_bond
        bond2lp_mask = src_is_bond & dst_is_lp
        bond2bond_mask = src_is_bond & dst_is_bond

        def add_orbital_relation(src_type, rel_name, dst_type, mask, src_offset, dst_offset):
            edge_index = pos_orbital_interaction_idxs[:, mask].clone()
            edge_attr = interaction_preds[mask].clone()
            if edge_index.shape[1] > 0:
                edge_index[0] -= src_offset
                edge_index[1] -= dst_offset
            SIMG[src_type, rel_name, dst_type].edge_index = edge_index
            SIMG[src_type, rel_name, dst_type].edge_attr = edge_attr

        def add_reverse_relation(fwd_key, rev_key): # artificial, for better message-passing and gradient
            SIMG[rev_key].edge_index = torch.vstack((SIMG[fwd_key].edge_index[1], SIMG[fwd_key].edge_index[0]))
            # SIMG[rev_key].edge_attr = _ use node embeddings


        ###### predicted forward interactions
        add_orbital_relation(
            'lp', 'interacts_with', 'lp',
            lp2lp_mask,
            src_offset=lp_idx_offset,
            dst_offset=lp_idx_offset
        )

        add_orbital_relation(
            'lp', 'interacts_with', 'bond',
            lp2bond_mask,
            src_offset=lp_idx_offset,
            dst_offset=bond_idx_offset
        )

        add_orbital_relation(
            'bond', 'interacts_with', 'lp',
            bond2lp_mask,
            src_offset=bond_idx_offset,
            dst_offset=lp_idx_offset
        )

        add_orbital_relation(
            'bond', 'interacts_with', 'bond',
            bond2bond_mask,
            src_offset=bond_idx_offset,
            dst_offset=bond_idx_offset
        )


        ###### artifical reverse interactions (for better message passing and gradient)
        add_reverse_relation(
            ('lp', 'interacts_with', 'bond'),
            ('bond', 'rev_interacts_with', 'lp'),
        )

        add_reverse_relation(
            ('bond', 'interacts_with', 'lp'),
            ('lp', 'rev_interacts_with', 'bond'),
        )

        add_reverse_relation(
            ('lp', 'interacts_with', 'lp'),
            ('lp', 'rev_interacts_with', 'lp')
        )

        add_reverse_relation(
            ('bond', 'interacts_with', 'bond'),
            ('bond', 'rev_interacts_with', 'bond')
        )



        SIMG.symbols = SIMGinput.symbols




            
    if return_stereoelectronics:
        return SIMG, (node_node_interaction_probs, atom2bond_preds, node_preds, interaction_preds)
    else:
        return SIMG