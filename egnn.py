import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import utils
from typing import Union

"""
More or less the same as ehoogeboom besides conditioning on h in the EGNN
"""

class GCL(nn.Module):
    def __init__(self, input_nf: int, output_nf: int, hidden_nf: int, normalization_factor: float, agg_method: str,
                 edge_attr_dim: int=0, node_attr_dim: int=0, act_fn=nn.SiLU(), attention: bool=False):
        super().__init__()
        input_edge = input_nf * 2
        self.attention = attention
        self.normalization_factor = normalization_factor
        self.agg_method = agg_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + node_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        
    def edge_model(self, source: torch.Tensor, target: torch.Tensor,
                   edge_attr: Union[torch.Tensor, None], edge_mask: Union[torch.Tensor, None]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        source: [E,NF] node features of source nodes for all edges
        target: [E,NF] node features of destination nodes for all edges
        edge_attr: [E, EA] edge attributes
        edge_mask: [E,] ?

        returns:
        out: [E,NF] message * attention * mask (edge_feat)
        mij: [E,NF] message
        """
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        
        return out, mij

    def node_model(self, h: torch.Tensor, edge_index: torch.Tensor,
                   edge_feat: torch.Tensor, node_attr: Union[torch.Tensor, None]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: [V,NF] node features
        edge_index: [2,E]
        edge_feat: [E,NF] edge features
        node_attr: [V,NA] node attributes

        returns:
        out: [V,NF] updated node features
        agg: [V, 2*NF+NA]
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0),
                                   normalization_factor=self.normalization_factor,
                                   agg_method=self.agg_method)
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)
        out = h + self.node_mlp(agg)
        return out, agg
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr:Union[torch.Tensor, None]=None,
                node_attr: Union[torch.Tensor, None]=None, node_mask: Union[torch.Tensor, None]=None,
                edge_mask: Union[torch.Tensor, None]=None) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask) #type: ignore
        if node_attr is not None:
            print(node_attr.shape)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr) #type: ignore
        if node_mask is not None:
            h = h * node_mask
        
        return h, mij
    
class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf: int, normalization_factor: float, agg_method: str,
                edge_attr_dim: int=1, act_fn=nn.SiLU(), tanh: bool=False, coords_range: float=10.0):
        super().__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edge_attr_dim
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer
        )
        self.normalization_factor = normalization_factor
        self.agg_method = agg_method

    def coord_model(self, h: torch.Tensor, coord: torch.Tensor, edge_index: list[torch.Tensor],
                    coord_diff: torch.Tensor, edge_attr: Union[torch.Tensor, None], edge_mask: Union[torch.Tensor, None]
                    ) -> torch.Tensor:
        """
        h: [V, NF] node features
        coord: [V,3] coordinates
        edge_index: [2,E]
        coord_diff: [V*V, 3]? xi-xj for all i, j
        edge_attr: [E,EA]
        edge_mask: [E,]

        return: [V,3] updated coordinates
        """
        row, col = edge_index
        if edge_attr is not None:
            input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        else:
            input_tensor = torch.cat([h[row], h[col]], dim=1)

        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)

        if edge_mask is not None:
            trans = trans * edge_mask
        
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   agg_method=self.agg_method)
        
        coord = coord + agg

        return coord
    
    def forward(self, h: torch.Tensor, coord: torch.Tensor, edge_index: list[torch.Tensor],
                coord_diff: torch.Tensor, edge_attr: Union[torch.Tensor, None]=None,
                node_mask: Union[torch.Tensor, None]=None, edge_mask: Union[torch.Tensor, None]=None
                ) -> torch.Tensor:
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask

        return coord
    
class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf: int, edge_attr_dim: int=2, device='cpu', act_fn=nn.SiLU(), n_layers: int=2,
                 attention: bool=True, tanh: bool=False, coords_range: float=15., norm_constant: float=1, sin_embedding=None,
                 normalization_factor: float=100, agg_method: str='sum'):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_constant = norm_constant
        self.sin_embedding= sin_embedding
        self.normalization_factor = normalization_factor
        self.agg_method = agg_method

        for i in range(n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                              self.normalization_factor, self.agg_method, edge_attr_dim=edge_attr_dim,
                                              act_fn=act_fn, attention=attention))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, self.normalization_factor, self.agg_method,
                                                       edge_attr_dim, act_fn, tanh, self.coords_range_layer))
        
        self.to(self.device)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: list[torch.Tensor],
                node_mask: Union[torch.Tensor, None]=None, edge_mask: Union[torch.Tensor, None]=None,
                edge_attr: Union[torch.Tensor, None]=None) -> tuple[torch.Tensor, torch.Tensor]:
        dist, coord_diff = coord2diff(x, edge_index, self.norm_constant)

        if self.sin_embedding is not None:
            dist = self.sin_embedding(dist)
        
        if edge_attr is not None:
            edge_attr = torch.cat([dist, edge_attr], dim=-1)
        else:
            edge_attr = dist

        for i in range(self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask) #type: ignore
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask) #type: ignore

        if node_mask is not None:
            h = h * node_mask

        return h, x
    
class EGNN(nn.Module):
    def __init__(self, in_node_nf: int, in_edge_nf: int, hidden_nf: int, device='cpu', act_fn=nn.SiLU(),
                 n_layers: int=3, attention: bool=False, out_node_nf: Union[int, None]=None, tanh: bool=False,
                 coords_range: float=15, norm_constant: float=1, inv_sublayers: int=2, sin_embedding: bool=False,
                 normalization_factor: float=100, agg_method: str='sum', update_feature: bool=False):
        super().__init__()
        self.in_node_nf = in_node_nf + 1 #one for time
        if out_node_nf is None:
            self.out_node_nf = self.in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range / n_layers)
        self.normalization_factor = normalization_factor
        self.agg_method = agg_method
        self.update_feature = update_feature

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbedding()
            edge_attr_dim = self.sin_embedding.dim * 2 + in_edge_nf
        else:
            self.sin_embedding = None
            edge_attr_dim = 1 + in_edge_nf #one for distance
        
        self.embedding = nn.Linear(self.in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, self.out_node_nf)

        if in_edge_nf != 0:
            self.edge_embedding = nn.Linear(in_edge_nf, self.hidden_nf)
        
        for i in range(n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_attr_dim, device,
                                                               act_fn, inv_sublayers, attention,
                                                               tanh, coords_range, norm_constant,
                                                               self.sin_embedding, self.normalization_factor, self.agg_method))
            
        self.to(self.device)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: list[torch.Tensor],
                node_mask: Union[torch.Tensor, None]=None, edge_mask: Union[torch.Tensor, None]=None,
                edge_attr: Union[torch.Tensor, None]=None) -> tuple[torch.Tensor, torch.Tensor]:
        #h = self.embedding(h)

        h_emb = self.embedding(h)

        for i in range(self.n_layers):
            if self.update_feature:
                h_emb, x = self._modules["e_block_%d" % i](h_emb, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr) #type: ignore
            else:
                _, x = self._modules["e_block_%d" % i](h_emb, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=edge_attr) #type: ignore

        #if self.update_feature:
        #    h = self.embedding_out(h_emb)

        h_out = self.embedding_out(h_emb) if self.update_feature else h
        
        if node_mask is not None:
            h_out = h_out * node_mask

        return h_out, x

class SinusoidsEmbedding(nn.Module):
    def __init__(self, max_res: float=15., min_res: float=15. / 2000., div_factor: int=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb.detach()
        
def coord2diff(x: torch.Tensor, edge_index: list[torch.Tensor], norm_constant: float=1) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, dim=1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)

    return radial, coord_diff

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor,
                         num_segments: int, normalization_factor: float, agg_method: str) -> torch.Tensor:
    """
    result[i] = sum_(k: * -> i) (data[k]) / Z
    if 'mean', Z is the in_deg of the node (if in_deg = 0 -> 1)
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)

    if agg_method == 'sum':
        result = result / normalization_factor
    
    if agg_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm

    return result

class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf: int, in_edge_nf: int=0, n_dims: int=3, hidden_nf: int=64, 
                 device='cpu', act_fn=nn.SiLU(), n_layers: int=4, attention: bool=False,
                 condition_time: bool=True, tanh: bool=False, norm_constant: float=1,
                 inv_sublayers: int=2, sin_embedding: bool=False, normalization_factor: float=100,
                 agg_method: str='sum', update_feature: bool=False):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf, in_edge_nf, hidden_nf, device, act_fn, n_layers,
            attention, None, tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor, agg_method=agg_method,
            update_feature=update_feature
        )
        self.in_node_nf = in_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

        self.to(self.device)

    def forward(self, t: torch.Tensor, xh: torch.Tensor, node_mask: torch.Tensor,
                edge_mask: Union[torch.Tensor, None], edge_attr: Union[torch.Tensor, None]
                ) -> torch.Tensor:
        """
        t[B,] timestep of batch
        xh[B,V,NF] node features
        node_mask[B,V]
        edge_mask[B,V,V]
        edge_attr[B,V,V,EA]
        """

        batch_size, max_atoms, dims = xh.shape
        h_dims = dims - self.n_dims

        edges = self.get_adj_matrix(max_atoms, batch_size, self.device)
        edges = [x.to(self.device) for x in edges]

        node_mask = node_mask.view(batch_size * max_atoms, 1) #[B*V,1]

        if edge_mask is not None:
            edge_mask = edge_mask.view(batch_size * max_atoms * max_atoms, 1) #[B*V*V,1]
            
        xh = xh.view(batch_size * max_atoms, -1).clone() #[B*V,NF]
        x = xh[:, 0:self.n_dims].clone() #[B*V,3]

        if edge_attr is not None:
            in_edge_attr = edge_attr.view(batch_size * max_atoms * max_atoms, -1)

        """ x = x.view(batch_size, max_atoms, -1)
        node_mask = node_mask.view(batch_size, max_atoms)
        utils.assert_mean_zero_with_mask(x, node_mask)
        x = x.view(batch_size * max_atoms, -1)
        node_mask = node_mask.view(batch_size * max_atoms, 1) #[B*V,1] """

        if h_dims == 0:
            h = torch.ones(batch_size * max_atoms, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone() #[B*V,NF']
        
        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(batch_size, 1).repeat(1, max_atoms) #[B,V]
                h_time = h_time.view(batch_size * max_atoms, 1) #[B*V,1]
            h = torch.cat([h, h_time], dim=1) #[B*V,NF'+1]
        
        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=in_edge_attr) #[B*V,NF'+1 / 3]

        vel = (x_final - x) * node_mask #[B*V,3]

        if self.condition_time:
            h_final = h_final[:, :-1] #[B*V,NF']
        
        vel = vel.view(batch_size, max_atoms, -1) #[B,V,3]

        if torch.any(torch.isnan(vel)):
            print("Warning. NaN in update")
            vel = torch.zeros_like(vel)
        
        if node_mask is None:
            vel = utils.remove_mean(vel)
        else:
            vel = utils.remove_mean_with_mask(vel, node_mask.view(batch_size, max_atoms))
        
        return vel

    def get_adj_matrix(self, n_nodes: int, batch_size: int, device) -> list[torch.Tensor]:
        """
        Makes B Kn's: Connect i * n + j with i * n + k for all i in range(B) and k in range(n)
        TODO:
        Perhaps only connect atoms that are bonded or within a cutoff?
        """
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
