import torch
import numpy as np
from typing import Union

def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1).sum(dim=-1)

def remove_mean(x: torch.Tensor) -> torch.Tensor:
    """
    removes the mean from a given set of coordinates
    x: set of atomic coordinates for batch of molecules [B,V,3]
    returns: set of cog-centred coordinates for batch of molecules [B,V,3]
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

def assert_correctly_masked(x: torch.Tensor, node_mask: torch.Tensor):
    assert (x * (1 - node_mask.unsqueeze(-1))).abs().max().item() < 1e-4, 'Not properly masked'

def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    removes the mean from a given set of coordinates
    where molecules may have different number of atoms
    x: set of atomic coordinates for batch of molecules [B,V,3]
    node_mask [B,V]
    returns: set of cog-centred coordinates for batch of molecules
    """

    masked_max_abs_value = (x * (1 - node_mask.unsqueeze(-1))).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdim=True) #[B,1]

    mean = torch.sum(x, dim=1, keepdim=True) / N.unsqueeze(-1) #[B,1,F]
    x = x - mean * node_mask.unsqueeze(-1)

    return x

def assert_mean_zero(x: torch.Tensor):
    mean = torch.mean(x, dim=1, keepdim=True)
    assert mean.abs().max().item() < 1e-4

def assert_mean_zero_with_mask(x: torch.Tensor, node_mask: torch.Tensor, eps: float=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative error: {rel_error}'

def sample_center_gravity_zero_gaussian_with_mask(size: tuple[int, int, int], device: torch.device,
                                                  node_mask: torch.Tensor) -> torch.Tensor:
    """
    Samples set of cog-centred coordinates with given size and node_mask
    """
    
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask.unsqueeze(-1)

    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected

def random_rotation(x: torch.Tensor) -> torch.Tensor:
    """
    Randomly rotates a given set of coordinates
    """
    
    batch_size, num_atoms, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2
    Rx = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    theta = torch.rand(batch_size, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    Rx[:, 1:2, 1:2] = cos
    Rx[:, 1:2, 2:3] = sin
    Rx[:, 2:3, 1:2] = -sin
    Rx[:, 2:3, 2:3] = cos

    Ry = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    theta = torch.rand(batch_size, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    Ry[:, 0:1, 0:1] = cos
    Ry[:, 0:1, 2:3] = -sin
    Ry[:, 2:3, 0:1] = sin
    Ry[:, 2:3, 2:3] = cos

    Rz = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    theta = torch.rand(batch_size, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    Rz[:, 0:1, 0:1] = cos
    Rz[:, 0:1, 1:2] = sin
    Rz[:, 1:2, 0:1] = -sin
    Rz[:, 1:2, 1:2] = cos

    x = x.transpose(1, 2)
    x = torch.matmul(Rx, x)
    x = torch.matmul(Ry, x)
    x = torch.matmul(Rz, x)
    x = x.transpose(1, 2)

    return x.contiguous()

class EMA():
    def __init__(self, beta: float=0.999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: torch.nn.Module, current_model: torch.nn.Module):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, new_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, new_weight)

    def update_average(self, old: Union[torch.Tensor, None], new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1. - self.beta) * new
    
class Queue():
    def __init__(self, max_len: int=50):
        self.items = []
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.items)
    
    def add(self, item: float) -> None:
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self) -> float:
        return np.mean(self.items) #type: ignore
    
    def std(self) -> float:
        return np.std(self.items) #type: ignore
    
def gradient_clipping(model, gradnorm_queue: Queue) -> torch.Tensor:
    max_grad_norm = 1.5 * gradnorm_queue.mean() + 2 * gradnorm_queue.std()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm, norm_type=2.0
    )

    if float(grad_norm) > max_grad_norm:
        gradnorm_queue.add(float(max_grad_norm))
    else:
        gradnorm_queue.add(float(grad_norm))

    # if float(grad_norm) > max_grad_norm:
    #     print(f'Clipped gradient with value {grad_norm:.1f} '
    #           f'while allowed {max_grad_norm:.1f}')
        
    return grad_norm