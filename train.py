import torch
from torch.utils.data import DataLoader, random_split
import dataset
from dataset import collate_fn
import diffusion
import copy
import utils
import os
import json
import argparse
from typing import Any

def str2bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description='set up hyperparameters')

    parser.add_argument('--exp_name', type=str, required=True, help='Exp name')
    parser.add_argument('--data_path', type=str, required=True, help='File path to dataset')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of cpu cores')
    parser.add_argument('--num_epoch', type=int, default=3000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.9999, help='Beta value for EMA')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--test_epoch', type=int, default=10, help='Epoch number for testing')
    parser.add_argument('--clip_grad', type=str2bool, default=True, help='Clip gradients')
    parser.add_argument('--in_node_nf', type=int, default=12, help='Input node features excluding charge')
    parser.add_argument('--in_edge_nf', type=int, default=2, help='Input edge features')
    parser.add_argument('--hidden_nf', type=int, default=128, help='Hidden feature dim')
    parser.add_argument('--n_layers', type=int, default=9, help='Number of layers')
    parser.add_argument('--attention', type=str2bool, default=True, help='Use attention')
    parser.add_argument('--condition_time', type=str2bool, default=True, help='Condition on time')
    parser.add_argument('--tanh', type=str2bool, default=True, help='Use tanh')
    parser.add_argument('--norm_constant', type=float, default=1., help='Normalization constant in EGNN')
    parser.add_argument('--inv_sublayers', type=int, default=1, help='Number of inverse sublayers')
    parser.add_argument('--sin_embedding', type=str2bool, default=False, help='Use sin embedding')
    parser.add_argument('--normalization_factor', type=float, default=1., help='Normalization factor')
    parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps for diffusion')
    parser.add_argument('--noise_schedule', type=str, default='polynomial_2', help='Noise schedule')
    parser.add_argument('--noise_precision', type=float, default=1e-5, help='Noise precision')
    parser.add_argument('--loss_type', type=str, default='l2', help='Loss function type')
    parser.add_argument('--norm_values', type=tuple, default=(1, 1, 1), help='Normalization values')
    parser.add_argument('--include_charges', type=str2bool, default=False, help='Include charges')
    parser.add_argument('--lambda_dist', type=float, default=0.002, help='Distance loss multiplier')
    parser.add_argument('--resume', type=str2bool, default=False, help='Resuming training')
    parser.add_argument('--gpu', type=str2bool, default=True, help='Use gpus?') #perhaps this should not be a parameter
    parser.add_argument('--random_rot', type=str2bool, default=False, help='Random rotation of training data')
    parser.add_argument('--noise_aug', type=float, default=0., help='Augment training data with noise')
    parser.add_argument('--num_gpus', type=int, default=1, help='Data parallel') #DP doesn't work as intended? Don't use.

    args = parser.parse_args()

    hp = vars(args)

    return hp

def get_dataset(path: str, batch_size: int, num_workers: int, gpu: bool) -> dict[str, DataLoader]:
    batch_size_val = batch_size

    data = dataset.QM9Dataset(dir=path)
    num_data = len(data)

    train_size = int(0.8 * num_data)
    val_size = int(0.1 * num_data)
    test_size = num_data - train_size - val_size

    train_set, val_set, test_set = random_split(
        data,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    persistent = num_workers > 0

    if gpu:
        loader = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=True, persistent_workers=persistent),
            'val': DataLoader(val_set, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=True, persistent_workers=persistent),
            'test': DataLoader(test_set, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers, pin_memory=True, persistent_workers=persistent)
        }

    else:
        loader = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers),
            'val': DataLoader(val_set, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn, num_workers=num_workers),
            'test': DataLoader(test_set, batch_size=batch_size_val, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
        }

    return loader

def save_json(path: str, hp: dict) -> None:
    assert path.endswith('.json')
    with open(path, 'w') as file:
        json.dump(hp, file, indent=4)

    print(f'Hyperparameters saved to {path}')

def load_json(path: str) -> dict:
    assert path.endswith('.json')
    with open(path, 'r') as file:
        hp = json.load(file)

    return hp

def main():
    hp = parse_args()

    resume = hp['resume']
    exp_name = hp['exp_name']

    if resume:
        assert os.path.exists(f'outputs/{exp_name}'), f'outputs/{exp_name} does not exist!'

    else:
        try:
            os.makedirs('outputs')
        except OSError:
            pass

        try:
            os.makedirs('outputs/' + exp_name)
        except OSError:
            pass
    
    #dir = f'outputs/{exp_name}'
    json_file = f'outputs/{exp_name}/hp.json'

    if resume:
        hp = load_json(json_file)

        assert 'best_epoch_model' in hp, 'Bad file, cannot resume training'
        assert 'best_epoch_rma' in hp, 'Bad file, cannot resume training'
        assert 'best_nll_val' in hp, 'Bad file, cannot resume training'
        assert 'best_nll_test' in hp, 'Bad file, cannot resume training'
        assert 'best_rma_val' in hp, 'Bad file, cannot resume training'
        assert 'last_epoch' in hp, 'Bad file, cannot resume training'
        assert 'queue' in hp, 'Bad file, cannot resume training'

    else:
        save_json(json_file, hp)

    data_path = hp['data_path']
    num_workers = hp['num_workers']
    #num_workers = 0
    num_epoch = hp['num_epoch']
    lr = hp['lr']
    beta = hp['beta']
    batch_size = hp['batch_size']
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
    lambda_dist = hp['lambda_dist']
    clip_grad = hp['clip_grad']
    test_epoch = hp['test_epoch']
    gpu = hp['gpu']
    random_rot = hp['random_rot']
    noise_aug = hp['noise_aug']
    dp = hp['num_gpus']

    best_epoch_model = hp['best_epoch_model'] if resume else 0
    best_epoch_rma = hp['best_epoch_rma'] if resume else 0
    best_nll_val = hp['best_nll_val'] if resume else 1e8
    best_nll_test = hp['best_nll_test'] if resume else 1e8
    best_rma_val = hp['best_rma_val'] if resume else 1e8
    last_epoch = hp['last_epoch'] if resume else 0

    loader = get_dataset(data_path, batch_size, num_workers, gpu)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = diffusion.EnVariationalDiffusion(in_node_nf_no_charge=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf,
                                            device=device, n_layers=n_layers, attention=attention,
                                            tanh=tanh, inv_sublayers=inv_sublayers, normalization_factor=normalization_factor,
                                            timesteps=timesteps, noise_schedule=noise_schedule, noise_precision=noise_precision,
                                            loss_type=loss_type, norm_values=norm_values, include_charges=include_charge,
                                            condition_time=condition_time, norm_constant=norm_constant, sin_embedding=sin_embedding)
    model.to(device)

    graph_nf = 2 + in_node_nf + include_charge + in_edge_nf

    rma_estimator = diffusion.ScalePredictor(graph_nf, hidden_nf, device)
    rma_estimator.to(device)

    if dp > 1 and gpu:
        model = torch.nn.DataParallel(model.cpu(), device_ids=list(range(dp))).cuda()
        rma_estimator = torch.nn.DataParallel(rma_estimator.cpu(), device_ids=list(range(dp))).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=lr,
                            amsgrad=True, weight_decay=1e-12)
    optim_rma = torch.optim.Adam(rma_estimator.parameters(), lr=lr)

    gradnorm_queue = utils.Queue()

    if beta > 0:
        model_ema = copy.deepcopy(model)
        ema = utils.EMA(beta)

    else:
        model_ema = model

    if resume:
        for grad in hp['queue']:
            gradnorm_queue.add(grad)

        model.load_state_dict(torch.load(f'outputs/{exp_name}/model_{last_epoch}.pth', map_location=device))
        rma_estimator.load_state_dict(torch.load(f'outputs/{exp_name}/rma_{last_epoch}.pth', map_location=device))
        optim.load_state_dict(torch.load(f'outputs/{exp_name}/optim_{last_epoch}.pth', map_location=device))
        optim_rma.load_state_dict(torch.load(f'outputs/{exp_name}/optim_rma_{last_epoch}.pth', map_location=device))
        if beta > 0:
            model_ema.load_state_dict(torch.load(f'outputs/{exp_name}/model_ema_{last_epoch}.pth', map_location=device))
        else:
            model_ema = model

    else:
        gradnorm_queue.add(3000)

    print("Data succesfully loaded")

    for epoch in range(last_epoch + 1, num_epoch + 1):
        model.train()
        nll_train_epoch = 0
        dist_loss_epoch = 0
        rma_loss_epoch = 0
        n_samples_train = 0
        for batch in loader['train']:
            x = batch['x'].to(device)
            node_attr = batch['node_attr'].to(device)
            charge = batch['charge'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            node_mask = batch['node_mask'].to(device)
            edge_mask = batch['edge_mask'].to(device)
            num_samples = edge_attr.size(0)

            if include_charge:
                h = torch.cat([node_attr, charge], dim=-1)
            else:
                h = node_attr

            optim_rma.zero_grad()
            rma_loss = diffusion.compute_rma_loss(x, node_attr, node_mask, edge_attr, edge_mask, rma_estimator)
            rma_loss = rma_loss.mean()
            rma_loss.backward()
            optim_rma.step()

            eps = utils.sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * noise_aug

            if random_rot:
                x = utils.random_rotation(x).detach()

            optim.zero_grad()
            loss, loss_dict = model(x, h, node_mask, edge_mask, edge_attr, lambda_dist)
            loss = loss.mean()
            loss.backward()

            if clip_grad:
                grad_norm = utils.gradient_clipping(model, gradnorm_queue)
            else:
                grad_norm = 0.

            optim.step()

            if beta > 0:
                ema.update_model_average(model_ema, model)

            nll_train_epoch += loss.item() * num_samples
            dist_loss_epoch += loss_dict['dist_loss'].mean().item() * num_samples
            rma_loss_epoch += rma_loss.item() * num_samples
            n_samples_train += num_samples

        print(f'Epoch: {epoch:>4} || Train Epoch NLL: {nll_train_epoch / n_samples_train:.10f} || Dist loss: {dist_loss_epoch / n_samples_train:.10f} || RMA loss: {rma_loss_epoch / n_samples_train:.10f}')

        torch.save(optim.state_dict(), 'outputs/%s/optim_%d.pth' % (exp_name, epoch))
        torch.save(optim_rma.state_dict(), 'outputs/%s/optim_rma_%d.pth' % (exp_name, epoch))
        torch.save(model.state_dict(), 'outputs/%s/model_%d.pth' % (exp_name, epoch))
        torch.save(rma_estimator.state_dict(), 'outputs/%s/rma_%d.pth' % (exp_name, epoch))

        if beta >0 :
            torch.save(model_ema.state_dict(), 'outputs/%s/model_ema_%d.pth' % (exp_name, epoch))

        if epoch % test_epoch == 0:
            model_ema.eval()
            rma_estimator.eval()

            nll_epoch_val = 0
            rma_epoch_val = 0
            n_samples_val = 0
            nll_epoch_test = 0
            rma_epoch_test = 0
            n_samples_test = 0

            with torch.no_grad():
                for batch in loader['val']:
                    x = batch['x'].to(device)
                    node_attr = batch['node_attr'].to(device)
                    charge = batch['charge'].to(device)
                    edge_attr = batch['edge_attr'].to(device)
                    node_mask = batch['node_mask'].to(device)
                    edge_mask = batch['edge_mask'].to(device)

                    num_samples = node_attr.size(0)
                    
                    if include_charge:
                        h = torch.cat([node_attr, charge], dim=-1)
                    else:
                        h = node_attr

                    rma_loss = diffusion.compute_rma_loss(x, node_attr, node_mask, edge_attr, edge_mask, rma_estimator)
                    rma_loss = rma_loss.mean()
                    rma_epoch_val += rma_loss.item() * num_samples

                    nll, _ = model_ema(x, h, node_mask, edge_mask, edge_attr)
                    nll = nll.mean()

                    nll_epoch_val += nll.item() * num_samples
                    n_samples_val += num_samples

                for batch in loader['test']:
                    x = batch['x'].to(device)
                    node_attr = batch['node_attr'].to(device)
                    charge = batch['charge'].to(device)
                    edge_attr = batch['edge_attr'].to(device)
                    node_mask = batch['node_mask'].to(device)
                    edge_mask = batch['edge_mask'].to(device)
                    num_samples = node_attr.size(0)
                    
                    if include_charge:
                        h = torch.cat([node_attr, charge], dim=-1)
                    else:
                        h = node_attr

                    rma_loss = diffusion.compute_rma_loss(x, node_attr, node_mask, edge_attr, edge_mask, rma_estimator)
                    rma_loss = rma_loss.mean()
                    rma_epoch_test += rma_loss.item() * num_samples

                    nll, _ = model_ema(x, h, node_mask, edge_mask, edge_attr)
                    nll = nll.mean()

                    nll_epoch_test += nll.item() * num_samples
                    n_samples_test += num_samples
                    
            avg_nll_test = nll_epoch_test / n_samples_test
            avg_rma_test = rma_epoch_test / n_samples_test
            avg_nll_val = nll_epoch_val / n_samples_val
            avg_rma_val = rma_epoch_val / n_samples_val

            if avg_rma_val < best_rma_val:
                best_rma_val = avg_rma_val
                best_epoch_rma = epoch

                torch.save(optim_rma.state_dict(), 'outputs/%s/optim_rma.pth' % exp_name)
                torch.save(rma_estimator.state_dict(), 'outputs/%s/rma.pth' % exp_name)


            if avg_nll_val < best_nll_val:
                best_nll_val = avg_nll_val
                best_nll_test = avg_nll_test
                best_epoch_model = epoch

                torch.save(optim.state_dict(), 'outputs/%s/optim.pth' % exp_name)
                #torch.save(optim.state_dict(), 'outputs/%s/optim_%d.pth' % (exp_name, epoch))
                torch.save(model.state_dict(), 'outputs/%s/model.pth' % exp_name)
                #torch.save(model.state_dict(), 'outputs/%s/model_%d.pth' % (exp_name, epoch))

                if beta > 0:
                    torch.save(model_ema.state_dict(), 'outputs/%s/model_ema.pth' % exp_name)
                    #torch.save(model_ema.state_dict(), 'outputs/%s/model_ema_%d.pth' % (exp_name, epoch))

            print('Val loss per sample: %.4f || Test loss per sample: %.4f' % (avg_nll_val, avg_nll_test))
            print('Val RMA loss per sample: %.4f || Test RMA loss per sample: %.4f' % (avg_rma_val, avg_rma_test))
            print('Best Model Epoch: %d || Best val loss per sample: %.4f || Best test loss per sample: %.4f' % (best_epoch_model, best_nll_val, best_nll_test))
            print('Best RMA Epoch: %d || Best val RMA loss per sample: %.4f' % (best_epoch_rma, best_rma_val))

            hp['best_epoch_model'] = best_epoch_model
            hp['best_epoch_rma'] = best_epoch_rma
            hp['best_nll_val'] = best_nll_val
            hp['best_nll_test'] = best_nll_test
            hp['best_rma_val'] = best_rma_val
            hp['last_epoch'] = epoch
            hp['queue'] = gradnorm_queue.items

            save_json(json_file, hp)

    assert best_epoch_model != 0, f'{best_epoch_model} cannot be best epoch'
    assert best_epoch_rma != 0, f'{best_epoch_rma} cannot be best epoch'

    save_json(json_file, hp)

if __name__ == "__main__":
    main()