import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import argparse
import ast
    
# the dataset can be downloaded from
# https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help
# nmrshiftdb2withsignals.sd


molsuppl = Chem.SDMolSupplier('./data/nmrshiftdb2withsignals.sd', removeHs = False)

atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
charge_list = [1, 2, 3, -1, -2, -3, 0]
degree_list = [1, 2, 3, 4, 5, 6, 0]
valence_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]

bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
max_graph_distance = 20

rdBase.DisableLog('rdApp.error') 
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

def get_atom_shifts_13C(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}
    for key in molprops.keys():
    
        if key.startswith('Spectrum 13C'):
            
            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)
            
                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []
                atom_shifts[shift_idx].append(shift_val)

    for j in range(mol.GetNumAtoms()):
        if j in atom_shifts:
            atom_shifts[j] = np.median(atom_shifts[j])

    return atom_shifts


def get_atom_shifts_1H(mol):
    
    molprops = mol.GetPropsAsDict()
    
    atom_shifts = {}

    for key in molprops.keys():
        
        if key.startswith('Spectrum 1H'):
            
            tmp_dict = {}

            for shift in molprops[key].split('|')[:-1]:
            
                [shift_val, _, shift_idx] = shift.split(';')
                shift_val, shift_idx = float(shift_val), int(shift_idx)

                if shift_idx not in atom_shifts: atom_shifts[shift_idx] = []

                if shift_idx not in tmp_dict: tmp_dict[shift_idx] = []
                tmp_dict[shift_idx].append(shift_val)

            for shift_idx in tmp_dict.keys():
                atom_shifts[shift_idx].append(tmp_dict[shift_idx])

    for shift_idx in atom_shifts.keys():
        max_len = np.max([len(shifts) for shifts in atom_shifts[shift_idx]])
        
        for i in range(len(atom_shifts[shift_idx])):

            if len(atom_shifts[shift_idx][i]) < max_len:

                if len(atom_shifts[shift_idx][i]) == 1:
                    atom_shifts[shift_idx][i] = [atom_shifts[shift_idx][i][0] for _ in range(max_len)]

                elif len(atom_shifts[shift_idx][i]) > 1:
                    while len(atom_shifts[shift_idx][i]) < max_len:
                        atom_shifts[shift_idx][i].append(np.mean(atom_shifts[shift_idx][i]))

            atom_shifts[shift_idx][i] = sorted(atom_shifts[shift_idx][i])

        atom_shifts[shift_idx] = np.median(atom_shifts[shift_idx], 0).tolist()
    

    return atom_shifts


def _DA(mol):

        D_list, A_list = [], []
        for feat in chem_feature_factory.GetFeaturesForMol(mol):
            if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
            if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
        
        return D_list, A_list

def _chirality(atom):

    if atom.HasProp('Chirality'):
        c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
    else:
        c_list = [0, 0]

    return c_list
    

def _stereochemistry(bond):

    if bond.HasProp('Stereochemistry'):
        s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
    else:
        s_list = [0, 0]

    return s_list    

def add_mol_sparsified_graph(mol_dict, mol):

    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)
    
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    
    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
        
        edge_attr = np.array(np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1), dtype = bool)
        edge_attr = np.vstack([edge_attr, edge_attr])
        
        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict

def add_mol_fully_connected_graph(mol_dict, mol):

    def _edge_features(bid1, bid2, mol, rings):
        atom_idx = [a.GetIdx() for a in mol.GetAtoms()]
        
        bondpath = Chem.GetShortestPath(mol, bid1, bid2)
        bonds = [mol.GetBondBetweenAtoms(bondpath[t], bondpath[t + 1]) for t in range(len(bondpath) - 1)]

        samering = 0
        for ring in rings:
            if bid1 in ring and bid2 in ring:
                samering = 1

        if len(bonds)==1:
            b = mol.GetBondBetweenAtoms(atom_idx[bid1], atom_idx[bid2])
            edge_fea1 = np.eye(len(bond_list), dtype = bool)[bond_list.index(str(b.GetBondType()))]
            edge_fea2 = np.array(_stereochemistry(b), dtype = bool)
            edge_fea3 = np.array([b.IsInRing(), b.GetIsConjugated()])
        else:
            edge_fea1 = np.zeros(4)
            edge_fea2 = np.zeros(2)
            edge_fea3 = np.zeros(2)

        edge_fea4 = np.eye(max_graph_distance, dtype = bool)[int(np.clip(len(bonds),0,max_graph_distance)-1)]
        edge_fea5 = np.array([samering])
        
            
        return np.array(np.concatenate([edge_fea1, edge_fea2, edge_fea3, edge_fea4, edge_fea5], axis=0), dtype = bool)

    n_node = mol.GetNumAtoms()
    n_edge = n_node * (n_node -1)

    D_list, A_list = _DA(mol)
    rings = mol.GetRingInfo().AtomRings()
    
    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])
    
    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    

    if n_edge > 0:

        edge_attr = []
        edge_loc = [[],[]]
        for j in range(n_node):
            for k in range(j+1, n_node):
                edge_attr.append(_edge_features(j, k, mol, rings))
                edge_loc[0].append(j)
                edge_loc[1].append(k)
        edge_attr = np.vstack([edge_attr, edge_attr])
        edge_loc = np.array(edge_loc)
        src = np.hstack([edge_loc[0,:], edge_loc[1,:]])
        dst = np.hstack([edge_loc[1,:], edge_loc[0,:]])
        
        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)
    
    return mol_dict





def preprocess(args):
    length = len(molsuppl)
    target = args.target
    graph_representation = args.graph_representation

    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': []}
               
    for i, mol in enumerate(molsuppl):

        try:
            Chem.SanitizeMol(mol)
            si = Chem.FindPotentialStereo(mol)
            for element in si:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                    mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                    mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
            assert '.' not in Chem.MolToSmiles(mol)
        except:
            continue

        if target == '13C': 
            atom_shifts = get_atom_shifts_13C(mol)
        elif target == '1H':
            atom_shifts = get_atom_shifts_1H(mol)


        
        if len(atom_shifts) == 0: continue
        for j, atom in enumerate(mol.GetAtoms()):
            if j in atom_shifts:
                atom.SetProp('shift', str(atom_shifts[j]))
                atom.SetBoolProp('mask', True)
            else:
                if target == '13C': atom.SetProp('shift', str(0))
                elif target == '1H': atom.SetProp('shift', str([0]))
                atom.SetBoolProp('mask', False)

        
        mol = Chem.RemoveHs(mol)

        if graph_representation == 'sparsified':
            mol_dict = add_mol_sparsified_graph(mol_dict, mol)

        elif graph_representation == 'fully_connected':
            mol_dict = add_mol_fully_connected_graph(mol_dict, mol)

        
        if (i+1)%1000==0: 
            print(f'{i+1}/{length} processed')

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    if target == '13C': mol_dict['shift'] = np.hstack(mol_dict['shift'])
    if target == '1H': mol_dict['shift'] = np.array(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])

    for key in mol_dict.keys(): print(key, mol_dict[key].shape, mol_dict[key].dtype)
        
    np.savez_compressed('./data/nmrshiftdb2_graph_%s_%s.npz'%(graph_representation, target), data = [mol_dict])

    

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--target', choices = ['13C', '1H'], type = str)
    arg_parser.add_argument('--graph_representation', choices = ['sparsified', 'fully_connected'], type = str)

    args = arg_parser.parse_args()
    
    preprocess(args)
