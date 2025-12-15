import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
import utils
import openbabel
from openbabel import openbabel as ob

ob.obErrorLog.SetOutputLevel(0) #Probably bad practice. To ignore Kekulize errors

dataset_info = {
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_num': {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F']
}

class QM9Dataset(Dataset):
    """
    keys for dataset dict
    input = N x VF tensor of node features and N*N x EF tensor of edge features
    output = N x 3 tensor of atomic position
    """
    def __init__(self, dir: str, norm_val: tuple[float, float, float, float]=(3, 4, 4, 10)):
        """
        dir(str) : path to directory containing xyz files
        norm_val(tuple of float) : normalization factors for bond order, degree, valency, atomic number
        """
        self.dir = dir
        self.file_paths = self.get_xyz_files(dir)
        self.atom_dict = dataset_info['atom_encoder'] #type: dict[str, int]
        self.atom_num = dataset_info['atom_num'] #type: dict[str, int]
        self.norm_val = norm_val

    def get_xyz_files(self, dir: str) -> list[str]:
        """
        dir(str) : path to directory containing xyz files
        returns: list(str) list of paths to the xyz files
        """
        file_paths = []
        for file in os.listdir(dir):
            full_file_path = os.path.join(dir, file)
            if os.path.isfile(full_file_path) and file.endswith('.xyz'):
                file_paths.append(full_file_path)
        
        return file_paths

    def get_atom_onehot(self, atom_type: str) -> np.ndarray:
        return self.get_onehot(5, self.atom_dict[atom_type])
    
    def get_onehot(self, num_classes: int, idx: int) -> np.ndarray:
        """
        num_classes(int): number of categories
        idx(int): index to be set to one
        return(ndarray): onehot vector of length num_classes with idx set to one
        """
        return np.eye(num_classes, dtype=np.long)[idx]

    def __len__(self) -> int:
        return len(self.file_paths)
    
    def load_xyz_file(self, file: str) -> tuple[list[str], np.ndarray, np.ndarray, ob.OBMol]:
        """
        parses xyz file to return atom types, coordinates, and charges,
        along with an OBMol object constructed from the xyz file

        file(str): path to an xyz file
        returns:
            atoms: list of atom symbol [N,]
            coords: numpy array of positions [N, 3]
            chrages: numpy array of partial charges(mulliken) [N,]
            mol: OBMol object constructed from xyz file
        """

        conv = ob.OBConversion()
        conv.SetInFormat('xyz')
        mol = ob.OBMol()
        conv.ReadFile(mol, file)

        with open(file, "r") as f:
            lines = f.readlines()

        num_atoms = int(lines[0].strip())
        atom_info = lines[2 : 2 + num_atoms]
        atoms = []
        coords = []
        charges = []

        #print(file)
        for atom in atom_info:
            info = atom.split()
            atoms.append(info[0])
            coords.append(list(map(lambda x: float(x.replace('*^-', 'e-')), info[1 : -1])))
            charges.append(float(info[-1].replace('*^-', 'e-')))

        return atoms, np.array(coords, dtype=np.float32), np.array(charges, dtype=np.float32), mol

    def parse_mol(self, mol: ob.OBMol) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        parses OBMol object to get atom/bond aromaticity; bond order;
        and atom degree, valency, and number

        mol(OBMol): OBMol object to be parsed
        returns:
            atom_ar: 1/0 encodings of atomic aromaticity [N,]
            bond_ar: 1/0 encodings of bond aromaticity [N,N]
            order: bond order [N,N]
            deg: degree(number of connected atoms) of each atom [N,]
            val: valency(sum of bond order) of each atom [N,]
            hyb: hybridization of each atom [N,]
        """
        num_atoms = mol.NumAtoms()
        atom_ar = np.zeros(num_atoms, dtype=np.long)
        bond_ar = np.zeros((num_atoms, num_atoms), dtype=np.long)
        order = np.zeros((num_atoms, num_atoms), dtype=np.float32)
        deg = np.zeros(num_atoms, dtype=np.float32)
        val = np.zeros(num_atoms, dtype=np.float32)
        hyb = np.zeros(num_atoms, dtype=np.long)
        
        for i in range(num_atoms):
            atom = mol.GetAtomById(i)
            deg[i] = atom.GetTotalDegree()
            val[i] = atom.GetTotalValence()
            hyb[i] = atom.GetHyb()
            if atom.IsAromatic():
                atom_ar[i] = 1

        for i in range(num_atoms):
            atom_i = mol.GetAtomById(i)
            for j in range(num_atoms):
                if i == j:
                    continue
                atom_j = mol.GetAtomById(j)

                bond = mol.GetBond(atom_i, atom_j)
                if bond is None:
                    continue

                order[i, j] = bond.GetBondOrder()
                if bond.IsAromatic():
                    bond_ar[i, j] = 1        

        return atom_ar, bond_ar, order, deg, val, hyb
    
    def normalize(self, order: np.ndarray, deg: np.ndarray, val: np.ndarray,
                  num: np.ndarray, norm_val: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        normalizes the input numpy arrays by the given normalization factors

        order: 2d numpy aray of bond order [N,N]
        deg: numpy array of atom degree [N,]
        val: numpy array of atom valency [N,]
        num: numpy array of atomic number [N,]
        norm_val: normalization factors for the four data

        returns: normalized values
        """
        return order / norm_val[0], deg / norm_val[1], val / norm_val[2], num / norm_val[3]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Key information:
        x: 3D coordinates of atoms in molecule [N,3]
        node_attr: attributes of atoms: atom type and hybridization one hot encodings,
                   1/0 encoding of atom aromaticity, normalized degree, valency, and atom number [N,12]
        edge_attr: bond attributes: normalized bond order and 1/0 encoding of aromaticity [N,N,2]
        charge: partial charges of atoms [N,1]
        """
        sample = dict()
        file = self.file_paths[idx]
        
        atoms, coords, charges, mol = self.load_xyz_file(file)
        #edges = infer_bonds(atoms, coords)

        atom_ar, bond_ar, order, deg, val, hyb = self.parse_mol(mol)

        num = np.array([self.atom_num[atom] for atom in atoms], dtype=np.float32)

        order, deg, val, num = self.normalize(order, deg, val, num, self.norm_val)

        onehot = torch.stack([torch.from_numpy(self.get_atom_onehot(atom)) for atom in atoms]) #[V,5]
        num = torch.from_numpy(num) #[V,]
        coords = torch.from_numpy(coords) #[V,3]
        coords = utils.remove_mean(coords.unsqueeze(0)).squeeze(0) #[V,3]
        charges = torch.from_numpy(charges) #[V,]
        atom_ar = torch.from_numpy(atom_ar) #[V,]
        bond_ar = torch.from_numpy(bond_ar) #[V,V]
        order = torch.from_numpy(order) #[V,V]
        deg = torch.from_numpy(deg) #[V,]
        val = torch.from_numpy(val) #[V,]
        hyb = torch.stack([torch.from_numpy(self.get_onehot(3, idx - 1)) for idx in hyb]) #[V,3]

        node_attr = torch.cat([onehot, hyb, torch.stack([atom_ar, deg, val, num], dim=-1)], dim=-1) #[V,5+3+1+1+1+1]
        edge_attr = torch.stack([order, bond_ar], dim=-1) #[V,V,1+1]

        sample['x'] = coords #[V,3]
        sample['node_attr'] = node_attr #[V,12] onehot + hyb + arom + deg + val + Z
        sample['edge_attr'] = edge_attr #[V,V,2] order + arom
        sample['charge'] = charges.unsqueeze(-1) #[V,1]

        return sample
    
def collate_fn(batch):
    """
    pads to
    x: [B,V,3]
    node_attr: [B,V,12]
    edge_attr: [B,V,V,2]
    charge: [B,V,1]
    where V is the max number of atoms
    """
    x = []
    node_attr = []
    edge_attr = []
    charges = []

    batch_size = len(batch)
    max_num = max(mol['x'].shape[0] for mol in batch)
    node_mask = torch.zeros((batch_size, max_num), dtype=torch.long)

    for i, mol in enumerate(batch):
        num_atoms = mol['x'].size(0)
        pad_size = max_num - num_atoms
        x.append(F.pad(mol['x'], (0, 0, 0, pad_size), "constant", 0))
        node_attr.append(F.pad(mol['node_attr'], (0, 0, 0, pad_size), "constant", 0))
        charges.append(F.pad(mol['charge'], (0, 0, 0, pad_size), "constant", 0))
        edge_attr.append(F.pad(mol['edge_attr'], (0, 0, 0, pad_size, 0, pad_size), "constant", 0))
        node_mask[i, :num_atoms] = 1

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    
    return {
        'x': torch.stack(x),
        'node_attr': torch.stack(node_attr),
        'edge_attr': torch.stack(edge_attr),
        'charge': torch.stack(charges),
        'node_mask': node_mask,
        'edge_mask': edge_mask,
    }

""" from torch.utils.data import DataLoader

#norm_val = (order, deg, val, Z)

dataset = QM9Dataset(dir='qm9/copy3')
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
max_bonds = dict()

print("Fully loaded") """

""" for batch in loader:
    edge_feat = batch['edge_feat'].squeeze() #[V,V,2]
    order = (edge_feat[:, :, 0] > 0).float() #[V,V]
    bonds = order.sum().item()

    if bonds not in max_bonds:
        max_bonds[bonds] = 0
    else:
        max_bonds[bonds] += 1

print(max_bonds) """
""" 
batch = next(iter(loader))

torch.set_printoptions(linewidth=200)

print('x')
print(batch['x'])
print('node_attr')
print(batch['node_attr'])
print('charge')
print(batch['charge'])
print('edge_attr')
print(batch['edge_attr'])
print('node_mask')
print(batch['node_mask'])
print('edge_mask')
print(batch['edge_mask']) """