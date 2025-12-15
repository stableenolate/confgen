import utils
import numpy as np
import math
import torch
from torch.nn import functional as F
import egnn
from typing import Union

"""
More or less the same as ehoogeboom except
1. The addition of RMA to normalize atomic coordinates
2. An addition of scale predictor to predict RMA from graph-wide attributes
3. Utility functions required for 1 and 2
"""

def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def clip_noise_schedule(alphas2: np.ndarray, clip_value: float=0.001) -> np.ndarray:
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = (alphas2[1:] / alphas2[:-1])
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s: float=1e-4, power: float=3.) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1. - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2)
    precision = 1 - 2 * s
    alphas2 = precision * alphas2 + s

    return alphas2

def cosine_beta_schedule(timesteps: int, s: float=0.008, raise_to_power: float=1.) -> np.ndarray:
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def gaussian_entropy(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(mu)

    return utils.sum_except_batch(
        zeros + 0.5 + torch.log(2 * np.pi * sigma ** 2) + 0.5
    )

def gaussian_KL(q_mu: torch.Tensor, q_sigma: torch.Tensor, p_mu: torch.Tensor,
                p_sigma: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    KL(q||p)
    """
    return utils.sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2)
            - 0.5
        ) * node_mask
    )

def gaussian_KL_for_dimension(q_mu: torch.Tensor, q_sigma: torch.Tensor,
                              p_mu: torch.Tensor, p_sigma: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    mu_norm2 = utils.sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1

    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm2) / (p_sigma ** 2) -0.5 * d

class PositiveLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool=True, weight_init_offset: float=-2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features))
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)
        
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)
    
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(100000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb
    
class PredefinedNoiseSchedule(torch.nn.Module):
    def __init__(self, noise_schedule: str, timesteps: int, precision: float):
        super().__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)
        
        #print('alphas2', alphas2)

        sigmas2 = 1. - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        #print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

class GammaNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps: int=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        #print('Gamma schedule:')
        #print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t: torch.Tensor) -> torch.Tensor:
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (gamma_tilde_1 - gamma_tilde_0)

        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma
    
def cdf_standard_gaussian(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

class ScalePredictor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_nf: int, device):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_nf),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_nf, hidden_nf),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_nf, 1)
        )

        self.to(device)

    def forward(self, graph_feat: torch.Tensor) -> torch.Tensor:
        """
        graph_feat[B,GF]
        return: [B,1,1]
        """
        return self.net(graph_feat).unsqueeze(-1)

class EnVariationalDiffusion(torch.nn.Module):
    def __init__(
            self, in_node_nf_no_charge: int, in_edge_nf: int, n_dims: int=3, hidden_nf: int=64,
            device='cpu', act_fn=torch.nn.SiLU(), n_layers=4, attention: bool=False, condition_time: bool=True,
            tanh: bool=False, norm_constant: float=1, inv_sublayers: int=2, sin_embedding: bool=False, 
            normalization_factor: float=100, agg_method: str='sum', timesteps: int=1000,
            noise_schedule: str='learned', noise_precision: float=1e-4, loss_type: str='vlb',
            norm_values: tuple[float, float, float]=(1., 1., 1.),
            norm_biases: tuple[Union[float, None], float, float]=(None, 0., 0.),
            include_charges: bool=True, update_feature: bool=False
    ):
        super().__init__()

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps, noise_precision)

        self.in_node_nf = in_node_nf_no_charge + include_charges
        if not condition_time:
            self.in_node_nf = self.in_node_nf - 1
        #self.in_edge_nf = in_edge_nf
        self.n_dims = n_dims
        #self.num_classes = self.in_node_nf
        #self.num_classes = num_classes
        self.cat_size = 8
        self.real_size = in_node_nf_no_charge - 8 + include_charges
        #assert self.num_classes == 5
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        self.device = device

        self.dynamics = egnn.EGNN_dynamics_QM9(self.in_node_nf, in_edge_nf, n_dims, hidden_nf, device, act_fn, n_layers,
                                               attention, condition_time, tanh, norm_constant, inv_sublayers,
                                               sin_embedding, normalization_factor, agg_method, update_feature)

        self.to(self.device)

        if noise_schedule != 'learned':
            self.check_issues_norm_values()
    
    def check_issues_norm_values(self, num_stdevs: float=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}'
            )

    def phi(self, xh: torch.Tensor, t: torch.Tensor, node_mask: torch.Tensor,
            edge_mask: Union[torch.Tensor, None], edge_attr: Union[torch.Tensor, None]) -> torch.Tensor:
        net_out = self.dynamics(t, xh, node_mask, edge_mask, edge_attr)
        return net_out
    
    def inflate_batch_array(self, array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    
    def sigma(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)
    
    def alpha(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    
    def SNR(self, gamma: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma)
    
    def subspace_dimensionality(self, node_mask: torch.Tensor) -> torch.Tensor:
        #print(node_mask.shape)
        number_of_nodes = torch.sum(node_mask, dim=1)
        return (number_of_nodes - 1) * self.n_dims
    
    def normalize(self, x: torch.Tensor, h: torch.Tensor, node_mask: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        rma = compute_rma(x, node_mask)

        x_norm = x / rma
        delta_log_px = delta_log_px - self.subspace_dimensionality(node_mask) * torch.log(rma.squeeze(-1).squeeze(-1))
        
        h_cat = h[:, :, :self.cat_size]
        h_real = h[:, :, self.cat_size:]

        assert h_real.size(-1) == self.real_size, f'{h.shape}, {self.cat_size}, {h_real.shape}'
        h_cat = (h_cat - self.norm_biases[1]) / self.norm_values[1] * node_mask.unsqueeze(-1)
        h_real = (h_real - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_real = h_real * node_mask.unsqueeze(-1)

        h = torch.cat([h_cat, h_real], dim=-1)

        return x_norm, h, delta_log_px, rma
    
    def unnormalize(self, x: torch.Tensor, h_cat: torch.Tensor, h_real: torch.Tensor,
                    node_mask: torch.Tensor, rma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x * self.norm_values[0]
        x = x * rma
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask.unsqueeze(-1)
        h_real = h_real * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_real = h_real * node_mask

        return x, h_cat, h_real
    
    def unnormalize_z(self, z: torch.Tensor, node_mask: torch.Tensor, rma: torch.Tensor) -> torch.Tensor:
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims + self.cat_size]
        h_real = z[:, :, self.n_dims + self.cat_size:]
        assert h_real.size(2) == self.real_size

        x, h_cat, h_real = self.unnormalize(x, h_cat, h_real, node_mask, rma)

        output = torch.cat([x, h_cat, h_real], dim=2)

        return output
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor,
                                  target_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s
    
    def kl_prior(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z1|x)||p(z1)=N(0,1))
        """
        ones = torch.ones((x.size(0), 1), device=x.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, x)

        mu_T = alpha_T * x

        sigma_T = self.sigma(gamma_T, mu_T)
        sigma_T = sigma_T.squeeze(-1).squeeze(-1)

        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(sigma_T)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T, sigma_T, zeros, ones, d=subspace_d)

        return kl_distance_x
    
    def compute_x_pred(self, net_out: torch.Tensor, xt: torch.Tensor, gamma_t: torch.Tensor) -> torch.Tensor:
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)

        eps_t = net_out
        x_pred = 1. / alpha_t * (xt - sigma_t * eps_t)

        return x_pred
    
    def compute_error(self, net_out: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            denom = self.n_dims * eps_t.shape[1]
            error = utils.sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = utils.sum_except_batch((eps - eps_t) ** 2)
        return error
    
    def log_constants_p_x_given_x0(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        p(x|x0)
        """
        batch_size = x.size(0)
        n_nodes = node_mask.sum(1)
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (-log_sigma_x - 0.5 * np.log(2 * np.pi))
    
    def sample_x_given_z0(self, z0: torch.Tensor, node_mask: torch.Tensor, edge_mask: Union[torch.Tensor, None],
                          edge_attr: Union[torch.Tensor, None], fix_noise: bool=False) -> torch.Tensor:
        """
        samples x ~ p(x|z0, G)
        """
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)

        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, edge_attr)
        x0 = z0[:, :, :self.n_dims]
        mu_x = self.compute_x_pred(net_out, x0, gamma_0)
        x = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        x = x * self.norm_values[0]

        return x
    
    def sample_normal(self, mu: torch.Tensor, sigma: torch.Tensor,
                      node_mask: torch.Tensor, fix_noise: bool=False) -> torch.Tensor:
        batch_size = 1 if fix_noise else mu.size(0)
        eps = self.sample_position_noise(batch_size, mu.size(1), node_mask)
        return mu + sigma * eps
    
    def sample_position_noise(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor) -> torch.Tensor:
        x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask
        )

        return x
    
    def log_px_given_z0_without_constants(self, eps: torch.Tensor, net_out: torch.Tensor) -> torch.Tensor:  
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_out, eps)

        return log_p_x_given_z_without_constants
    
    def compute_loss(self, x: torch.Tensor, h: torch.Tensor, node_mask: torch.Tensor,
                     edge_mask: Union[torch.Tensor, None], edge_attr: Union[torch.Tensor, None], t0_always: bool,
                     lambda_dist: float=0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if t0_always:
            lowest_t = 1
        else:
            lowest_t = 0

        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1),
            device=x.device
        ).float()

        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()

        s = s_int / self.T
        t = t_int / self.T

        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        eps = self.sample_position_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
        )

        xt = alpha_t * x + sigma_t * eps

        utils.assert_mean_zero_with_mask(xt, node_mask)
        #print(f'x: {xt.shape}, h: {h.shape}')
        zt = torch.cat([xt, h], dim=-1)
        net_out = self.phi(zt, t, node_mask, edge_mask, edge_attr)
        error = self.compute_error(net_out, eps)

        if edge_attr is not None:
            x_pred = self.compute_x_pred(net_out, xt, gamma_t)
            d_pred = torch.cdist(x_pred, x_pred, p=2)
            d_true = torch.cdist(x, x, p=2)

            order = edge_attr[:, :, :, 0]
            bond_mask = (order > 0).float()

            assert d_pred.shape == d_true.shape
            assert d_pred.shape == bond_mask.shape

            dist_loss = (((d_pred - d_true) ** 2) * bond_mask).sum(dim=(1, 2)) / bond_mask.sum(dim=(1, 2)) #[B,]

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        
        assert error.size() == SNR_weight.size()

        loss_t_larger_than_zero = 0.5 * SNR_weight * error
        neg_log_constants = -self.log_constants_p_x_given_x0(x, node_mask)
        #print(f'L0_const: {neg_log_constants}')

        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        kl_prior = self.kl_prior(x, node_mask)

        #print(f'kl: {kl_prior}, Lt: {loss_t_larger_than_zero}')

        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T
            estimator_loss_terms = num_terms * loss_t

            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            eps_0 = self.sample_position_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
            )

            x_0 = alpha_0 * x + sigma_0 * eps_0
            z_0 = torch.cat([x_0, h], dim=-1)
            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, edge_attr)

            loss_term_0 = -self.log_px_given_z0_without_constants(eps_0, net_out)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()
            assert kl_prior.size() == dist_loss.size()

            #print(f'reconstruction: {kl_prior}, Lt: {estimator_loss_terms}, L0: {loss_term_0 + neg_log_constants}')

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0 + dist_loss * lambda_dist

        else:
            loss_term_0 = -self.log_px_given_z0_without_constants(eps, net_out)

            #print(f'L0_wo_const: {loss_term_0}')

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == dist_loss.size()
            #print(estimator_loss_terms)

            loss = kl_prior + estimator_loss_terms + neg_log_constants + dist_loss * lambda_dist

            """ print(f'kl_prior: {kl_prior.item()}')
            print(f'estimator: {estimator_loss_terms.item()}')
            print(f'neg_log: {neg_log_constants.item()}') """

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(), 'error': error.squeeze(), 'dist_loss': dist_loss.squeeze()}
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, node_mask: torch.Tensor,
                edge_mask: torch.Tensor, edge_attr: torch.Tensor,
                lambda_dist: float=0.1) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        x, h, delta_log_px, _ = self.normalize(x, h, node_mask)

        utils.assert_correctly_masked(x, node_mask)
        utils.assert_correctly_masked(h, node_mask)
        if edge_attr is not None and edge_mask is not None:
            utils.assert_correctly_masked(edge_attr, edge_mask)

        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)

        if self.training:
            loss, loss_dict = self.compute_loss(
                x, h, node_mask, edge_mask, edge_attr, False, lambda_dist
            )

        else:
            loss, loss_dict = self.compute_loss(
                x, h, node_mask, edge_mask, edge_attr, True, lambda_dist
            )

        neg_log_px = loss

        assert neg_log_px.size() == delta_log_px.size()

        neg_log_px = neg_log_px - delta_log_px

        return neg_log_px, loss_dict
    
    def sample_p_xs_given_xt(self, s: torch.Tensor, t: torch.Tensor, xt: torch.Tensor,
                             h: torch.Tensor, node_mask: torch.Tensor, edge_mask: Union[torch.Tensor, None],
                             edge_attr: Union[torch.Tensor, None], fix_noise: bool=False) -> torch.Tensor:
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, xt)
        sigma_s = self.sigma(gamma_s, xt)
        sigma_t = self.sigma(gamma_t, xt)

        zt = torch.cat([xt, h], dim=-1)
        eps_t = self.phi(zt, t, node_mask, edge_mask, edge_attr)

        #print(f'eps_t: {eps_t}')

        utils.assert_mean_zero_with_mask(xt, node_mask)
        utils.assert_mean_zero_with_mask(eps_t, node_mask)

        mu = xt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        sigma = sigma_t_given_s * sigma_s / sigma_t

        xs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        #print(f'xs: {xs}')

        xs = utils.remove_mean_with_mask(xs, node_mask)

        return xs
    
    @torch.no_grad()
    def sample(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor, edge_mask: Union[torch.Tensor, None],
               h: torch.Tensor, edge_attr: Union[torch.Tensor, None], rma_estimator: ScalePredictor, fix_noise: bool=False) -> torch.Tensor:
        if fix_noise:
            xt = self.sample_position_noise(1, n_nodes, node_mask)
        else:
            xt = self.sample_position_noise(n_samples, n_nodes, node_mask)

        utils.assert_mean_zero_with_mask(xt, node_mask)

        for i in reversed(range(self.T)):
            s_array = torch.full((n_samples, 1), fill_value=i, device=xt.device)
            t_array = s_array + 1
            s = s_array / self.T
            t = t_array / self.T

            xt = self.sample_p_xs_given_xt(s, t, xt, h, node_mask, edge_mask, edge_attr, fix_noise)
        
        z0 = torch.cat([xt, h], dim=-1)
        x = self.sample_x_given_z0(z0, node_mask, edge_mask, edge_attr, fix_noise)

        utils.assert_mean_zero_with_mask(x, node_mask)

        graph_feat = get_graph_feat(h, node_mask, edge_mask, edge_attr)
        rma_est = rma_estimator(graph_feat)

        x = self.unnormalize_z(torch.cat([x, h], dim=-1), node_mask, rma_est)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()

        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting'
                  f'the positions down')
            
            x = utils.remove_mean_with_mask(x, node_mask)

        return x
    
    @torch.no_grad()
    def sample_chain(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor,
                    edge_mask: Union[torch.Tensor, None], h: torch.Tensor, edge_attr: Union[torch.Tensor, None],
                    rma_estimator: ScalePredictor, keep_frames:Union[int, None]=None) -> torch.Tensor:
        x = self.sample_position_noise(n_samples, n_nodes, node_mask)
        #x, h, _ = self.normalize(x, h, node_mask)
        utils.assert_mean_zero_with_mask(x, node_mask)
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        
        chain = torch.zeros((keep_frames,) + x.size(), device=x.device)

        graph_feat = get_graph_feat(h, node_mask, edge_mask, edge_attr)
        rma_est = rma_estimator(graph_feat)

        for i in reversed(range(self.T)):
            s_array = torch.full((n_samples, 1), fill_value=i, device=x.device)
            t_array = s_array + 1
            s = s_array / self.T
            t = t_array / self.T

            x = self.sample_p_xs_given_xt(s, t, x, h, node_mask, edge_mask, edge_attr)

            utils.assert_mean_zero_with_mask(x, node_mask)

            write_index = (i * keep_frames) // self.T
            z = torch.cat([x, h], dim=-1)
            z_unnorm = self.unnormalize_z(z, node_mask, rma_est)
            x_chain = z_unnorm[:, :, :self.n_dims]
            chain[write_index] = x_chain

        x = self.sample_x_given_z0(z, node_mask, edge_mask, edge_attr)
        utils.assert_mean_zero_with_mask(x, node_mask)
        z = torch.cat([x, h], dim=-1)
        z_unnorm = self.unnormalize_z(z, node_mask, rma_est)
        x_chain = z_unnorm[:, :, :self.n_dims]
        chain[0] = x_chain

        #chain_flat = chain.view(n_samples * keep_frames, *x.size()[1:])

        return chain

def compute_rma(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    r2 = (x ** 2).sum(dim=-1, keepdim=True) #[B,V,1]
    utils.assert_correctly_masked(r2, node_mask)
    N = utils.sum_except_batch(node_mask).unsqueeze(-1).unsqueeze(-1) #[B,1,1]
    r2_mean = (r2 * node_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (N + 1e-8) #[B,1,1]
    rma = torch.sqrt(r2_mean + 1e-8) #[B,1,1]

    return rma

def get_graph_feat(h: torch.Tensor, node_mask: torch.Tensor,
                    edge_mask: Union[torch.Tensor, None], edge_attr: Union[torch.Tensor, None]) -> torch.Tensor:
    batch_size, num_atoms, _ = h.shape
    V = node_mask.sum(dim=-1, keepdim=True) #[B,1]
    node_feat = h.sum(dim=1) / V #[B,NA]
    V_norm = V / 20.

    if edge_attr is not None and edge_mask is not None:
        bond_mask = (edge_attr[:, :, :, 0] > 0).float()
        
        E = bond_mask.view(batch_size, num_atoms * num_atoms).sum(dim=-1, keepdim=True) #[B,1]
        edge_feat = edge_attr.view(batch_size, num_atoms * num_atoms, -1).sum(dim=1) / E #[B,EA]
        E_norm = E / 40.
            
    graph_feat = torch.cat([V_norm, E_norm, node_feat, edge_feat], dim=-1) #[B,NA+EA+2]

    return graph_feat

def compute_rma_loss(x: torch.Tensor, h: torch.Tensor, node_mask: torch.Tensor,
                         edge_attr: torch.Tensor, edge_mask: torch.Tensor,
                         rma_estimator: Union[ScalePredictor, torch.nn.DataParallel[ScalePredictor]]) -> torch.Tensor:
    rma = compute_rma(x, node_mask) #[B,1,1]

    graph_feat = get_graph_feat(h, node_mask, edge_mask, edge_attr) #[B,GF=EA+NA]

    rma_est = rma_estimator(graph_feat) #[B,1,1]

    rma_loss = ((rma - rma_est) ** 2).squeeze(-1).squeeze(-1)

    return rma_loss #[B,]