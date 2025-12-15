import argparse
import json
import os
import torch
import diffusion
import dataset
import numpy as np
import torch.nn.functional as F
import openbabel
from openbabel import openbabel as ob

atom_encoder = dataset.dataset_info['atom_encoder']
atom_num = dataset.dataset_info['atom_num']

def str2bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

def parse_file(file_path: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parses molecular graph file and returns node and edge attributes

    file_path: string of path to txt file containing molecular graph
    returns:
        node_feat: normalized node attribute (atom type + hybridiztion + aromaticity + degree + valency + atomic number)
        edge_feat: normalized edge attributes (bond order + aromaticity)
    """
    
    onehot = []
    hyb = []
    other = []

    with open(file_path, 'r') as file:
        num_atoms = int(file.readline().strip())

        for _ in range(num_atoms):
            atom_data = file.readline().strip().split()
            assert len(atom_data) == 5, f'{file_path}, {atom_data} Bad input. Input must be Symb Hyb Arom Deg Val'

            atom_symbol = atom_data[0]
            hyb_idx = int(atom_data[1])
            
            if atom_symbol in atom_encoder:
                onehot.append(np.eye(5, dtype=np.int64)[atom_encoder[atom_symbol]])
            else:
                raise ValueError(f'Unknown symbol: {atom_symbol}')
            
            hyb.append(np.eye(3, dtype=np.int64)[hyb_idx - 1])
            arom, deg, val = map(np.float32, atom_data[2:])
            z = atom_num[atom_symbol]
            other.append(np.array([arom, deg / 4, val / 4, z / 10], dtype=np.float32))
            
        bond_feat = np.zeros((num_atoms, num_atoms, 2), dtype=np.float32)
        
        for line in file:
            assert len(line.split()) == 4
            atom_idx1, atom_idx2, bond_order, arom = map(np.int64, line.split())
            bond_feat[atom_idx1, atom_idx2, 0] = bond_order / 3.
            bond_feat[atom_idx2, atom_idx1, 0] = bond_order / 3.
            bond_feat[atom_idx1, atom_idx2, 1] = arom
            bond_feat[atom_idx2, atom_idx1, 1] = arom
    
    onehot = np.stack(onehot, axis=0)
    onehot = torch.from_numpy(onehot) #[V,5]
    hyb = np.stack(hyb, axis=0)
    hyb = torch.from_numpy(hyb) #[V,3]
    other = np.stack(other, axis=0)
    other = torch.from_numpy(other) #[V,4]

    node_feat = torch.cat([onehot, hyb, other], dim=-1).to(device) #[V,12]
    edge_feat = torch.from_numpy(bond_feat).to(device) #[V,V,2]

    return node_feat, edge_feat

def parse_dir(dir_path: str, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Parses directory containing molecular graph txt files and returns attributes, masks, and file names

    node_collate: collated node attributes [B,V,12]
    edge_collate: collated edge attributes [B,V,V,2]
    node_mask [B,V]
    edge_mask [B,V,V]
    file_names: list of file names without .txt extension. Used to keep track of file names when generating
    """
    node_batch = []
    edge_batch = []
    node_collate = []
    edge_collate = []
    file_names = []
    
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.txt'):
            file_names.append(file_name.split('.')[0])
            file_path = os.path.join(dir_path, file_name)
            node_feat, edge_feat = parse_file(file_path, device)
            node_batch.append(node_feat)
            edge_batch.append(edge_feat)
    
    batch_size = len(node_batch)
    max_num = max(node_attr.shape[0] for node_attr in node_batch)
    node_mask = torch.zeros((batch_size, max_num), dtype=torch.long).to(device)

    for i, (node_attr, edge_attr) in enumerate(zip(node_batch, edge_batch)):
        num_atoms = node_attr.size(0)
        pad_size = max_num - num_atoms
        node_collate.append(F.pad(node_attr, (0, 0, 0, pad_size), "constant", 0))
        edge_collate.append(F.pad(edge_attr, (0, 0, 0, pad_size, 0, pad_size), "constant", 0))
        node_mask[i, :num_atoms] = 1

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

    return torch.stack(node_collate), torch.stack(edge_collate), node_mask, edge_mask, file_names

def write_xyz(x: torch.Tensor, node_attr: torch.Tensor, node_mask: torch.Tensor,
              edge_attr: torch.Tensor, dir: str, file_names: list[str]) -> None:
    """
    Writes xyz files for each molecules in a
    batch into the given directory path(dir)
    """
    decoder = dataset.dataset_info['atom_decoder']
    batch_size, max_atoms, _ = x.shape
    valid_mol_cnt = 0
    N = torch.sum(node_mask, dim=1)

    if not os.path.exists(dir):
        os.makedirs(dir)

    for batch in range(batch_size):
        file_name = file_names[batch]
        num_atoms = N[batch].item()
        assert isinstance(num_atoms, int)

        mol_coords = x[batch, :num_atoms, :].cpu() #[N,3]
        mol_coords = [mol_coords[atom] for atom in range(num_atoms)]
        mol_onehot = node_attr[batch, :num_atoms, :5].cpu() #[N,5]
        atom_idx = torch.argmax(mol_onehot, dim=-1).cpu() #[N]
        atom_symbol = [decoder[idx] for idx in atom_idx]
        mol_order = edge_attr[batch, :num_atoms, :num_atoms, 0].cpu() #[N,N]

        output_file = os.path.join(dir, f'{file_name}.xyz')

        with open(output_file, 'w') as f:
            f.write(f'{num_atoms}\n')
            f.write(f'Generated by EDM\n')
            for symbol, coord in zip(atom_symbol, mol_coords):
                f.write(f'{symbol} {coord[0].item():.6f} {coord[1].item():.6f} {coord[2].item():.6f}\n')

        f.close()

        print(f'XYZ file saved to {output_file}')

        conv = ob.OBConversion()
        conv.SetInFormat('xyz')
        mol = ob.OBMol()
        conv.ReadFile(mol, output_file)

        bond_order_calc = torch.zeros((num_atoms, num_atoms))

        if mol is not None:
            for i in range(num_atoms):
                atom_i = mol.GetAtomById(i)
                for j in range(num_atoms):
                    if i == j:
                        continue
                    atom_j = mol.GetAtomById(j)
                    bond = mol.GetBond(atom_i, atom_j)
                    if bond is None:
                        continue
                    bond_order_calc[i, j] = bond.GetBondOrder()

            assert mol_order.shape == bond_order_calc.shape, f'{mol_order.shape} and {bond_order_calc.shape} do not match'

            if torch.equal(torch.round(mol_order * 3), bond_order_calc):
                valid_mol_cnt += 1
                print(f'{file_name} is OK')

            else:
                print(f'{file_name} has wrong bond order!') #Does not work properly due to ambiguous bond order in aromatic compounds
                print(f'Expected: {torch.round(mol_order * 3)}')
                print(f'Generated: {bond_order_calc}')
        else:
            print(f'{file_name} is invalid!')

    print("All molecules written")
    print(f'Validity: {valid_mol_cnt}/{batch_size}') #Not really reliable

def write_traj(x_chain: torch.Tensor, node_attr: torch.Tensor, node_mask: torch.Tensor,
               edge_attr: torch.Tensor, dir: str, file_names: list[str]) -> None:
    """
    Writes diffusion "trajectory" xyz files for each molecule in a
    given sampling chain into given directory path(dir)
    """
    
    decoder = dataset.dataset_info['atom_decoder']
    num_frames, batch_size, _, _ = x_chain.shape
    N = torch.sum(node_mask, dim=1)

    if not os.path.exists(dir):
        os.makedirs(dir)

    x_chain = x_chain.permute(1, 0, 2, 3)

    for batch in range(batch_size):
        file_name = file_names[batch]
        output_file = os.path.join(dir, f'{file_name}_traj.xyz')
        num_atoms = N[batch]

        mol_onehot = node_attr[batch, :num_atoms, :5].cpu()
        atom_idx = torch.argmax(mol_onehot, dim=-1).cpu()
        atom_symbol = [decoder[idx] for idx in atom_idx]

        with open(output_file, 'w') as f:
            for frame in reversed(range(num_frames)):
                f.write(f'{num_atoms}\n')
                f.write('\n')

                mol_coords = x_chain[batch, frame, :num_atoms, :].cpu()
                mol_coords = [mol_coords[atom] for atom in range(num_atoms)]
                
                for symbol, coord in zip(atom_symbol, mol_coords):
                    f.write(f'{symbol} {coord[0].item():.6f} {coord[1].item():.6f} {coord[2].item():.6f}\n')

                f.write('\n')
        
        f.close()

        print(f'XYZ trajectory saved to {output_file}')

def main():
    parser = argparse.ArgumentParser(description='Get path to hyperparameters and model parameters')
    parser.add_argument('--name', type=str, required=True, help='experiment name')
    parser.add_argument('--dir', type=str, required=True, help='directory that holds molecular graphs')
    parser.add_argument('--chain', type=str2bool, default=False, help='generate chain')
    parser.add_argument('--frames', type=int, default=None, help='number of frames')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    exp_name = args.name
    dir = args.dir
    chain = args.chain
    frames = args.frames

    node_attr, edge_attr, node_mask, edge_mask, file_names = parse_dir(dir, device)
    batch_size, num_atoms, _ = node_attr.shape

    json_path = f'outputs/{exp_name}/hp.json'

    with open(json_path, 'r') as json_file:
        hp = json.load(json_file)

    exp_name = hp['exp_name']
    assert exp_name == args.name
    beta = hp['beta']
    in_node_nf = hp['in_node_nf']
    in_edge_nf = hp['in_edge_nf']
    hidden_nf = hp['hidden_nf']
    n_layers = hp['n_layers']
    attention = hp['attention']
    condition_time = hp['condition_time']
    tanh = hp['tanh']
    norm_constant = hp['norm_constant']
    inv_sublayers = hp['inv_sublayers']
    sin_embedding = hp['sin_embedding']
    normalization_factor = hp['normalization_factor']
    timesteps = hp['timesteps']
    noise_schedule = hp['noise_schedule']
    noise_precision = hp['noise_precision']
    loss_type = hp['loss_type']
    norm_values = hp['norm_values']
    include_charge = hp['include_charges']
    gpu = hp['gpu']

    assert 'best_epoch_model' in hp and 'best_epoch_rma' in hp

    best_epoch_model = hp['best_epoch_model']
    best_epoch_rma = hp['best_epoch_rma']

    assert best_epoch_model != 0 and best_epoch_rma != 0

    print(f'Hyperparameters loaded from {json_path}')

    model = diffusion.EnVariationalDiffusion(in_node_nf_no_charge=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf,
                                            device=device, n_layers=n_layers, attention=attention,
                                            tanh=tanh, inv_sublayers=inv_sublayers, normalization_factor=normalization_factor,
                                            timesteps=timesteps, noise_schedule=noise_schedule, noise_precision=noise_precision,
                                            loss_type=loss_type, norm_values=norm_values, include_charges=include_charge,
                                            condition_time=condition_time, norm_constant=norm_constant, sin_embedding=sin_embedding)
    model.to(device)
    model.eval()

    graph_nf = 2 + in_node_nf + include_charge + in_edge_nf
    rma_estimator = diffusion.ScalePredictor(graph_nf, hidden_nf, device)
    rma_estimator.to(device)
    rma_estimator.eval()

    try:
        os.makedirs('outputs/' + exp_name + '/generated')
    except OSError:
        pass
    
    model_file_name = f'model_ema_{best_epoch_model}.pth' if beta > 0 else f'model_{best_epoch_model}.pth'
    model.load_state_dict(torch.load(f'outputs/{exp_name}/{model_file_name}', map_location=device))
    rma_file_name = f'rma_{best_epoch_rma}.pth'
    rma_estimator.load_state_dict(torch.load(f'outputs/{exp_name}/{rma_file_name}', map_location=device))

    model.eval()
    rma_estimator.eval()

    if not chain:
        x_sample = model.sample(batch_size, num_atoms,
                                node_mask, edge_mask, node_attr,
                                edge_attr, rma_estimator)
        
        write_xyz(x_sample, node_attr, node_mask, edge_attr, f'outputs/{exp_name}/generated', file_names)

    else:
        x_chain = model.sample_chain(batch_size, num_atoms,
                                     node_mask, edge_mask, node_attr,
                                     edge_attr, rma_estimator, frames)
        write_traj(x_chain, node_attr, node_mask, edge_attr, f'outputs/{exp_name}/generated', file_names)

if __name__ == "__main__":
    main()