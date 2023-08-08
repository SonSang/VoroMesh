from utils.custom_voronoi import Voroloss_opt
from networks.vc_decoders import VCDec
from networks.sdf_encoders import LocalSDFFE_ME, GlobFeatEnc
from utils.networks import cnt_params, AverageMeter, save_model
from datasets.sample_transformation import ComposeSampleTransformation
from datasets.cloud_transformation import ComposeCloudTransformation
from datasets.ABCDataset import ABCDataset, abc_collate_fn
import MinkowskiEngine as ME
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from time import time
import yaml
import io
import argparse
import os
os.environ['OMP_NUM_THREADS'] = '16'


def define_options_parser():
    parser = argparse.ArgumentParser(
        description='Model training script. Provide a suitable config.')
    parser.add_argument('config', type=str,
                        help='Path to config file in YAML format.')
    parger.add_argument('--reg_w', type=float, default=1.0,
                        help='Max offset regularization weight.')
    parger.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parger.add_argument('--lr', type=float,
                        default=0.000256, help='Learning rate.')
    parger.add_argument('--beta2', type=float, default=0.99,
                        help='Second moment weihght in adam optimizer.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--resume', action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')
    return parser


def training_step():
    data_time = AverageMeter()
    batch_time = AverageMeter()
    VL = AverageMeter()
    R = AverageMeter()

    encoder.train()
    if config['vcdec_in_glob']:
        glob_encoder.train()
    vcdec.train()
    torch.set_grad_enabled(True)

    end = time()
    for i, batch in enumerate(train_iterator):
        torch.cuda.empty_cache()
        if i >= len(train_iterator):
            break
        data_time.update(time() - end)
        end = time()

        input_sparse_xyz = batch['input_sparse_xyz'].to(device)
        input_sparse_sdf = (
            batch['input_sparse_sdf'].to(device) *
            torch.repeat_interleave(batch['input_gs'].to(device) / 32.,
                                    batch['input_sparse_sdf_idx_size'].to(
                                        device),
                                    dim=0)
        ).unsqueeze(1)
        input_sparse_sdf_idx = batch['input_sparse_sdf_idx'].to(device).int()
        input_sdf = ME.SparseTensor(input_sparse_sdf, input_sparse_sdf_idx)
        grid_features = encoder(input_sdf)

        if config['vcdec_in_glob'] and config['vcdec_p_dim']:
            vc = vcdec(
                grid_features.features,
                query=input_sparse_xyz,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(
                                                       device),
                                                   dim=0)
            )
        elif config['vcdec_in_glob']:
            vc = vcdec(
                grid_features.features,
                g_features=torch.repeat_interleave(glob_encoder(maxpooler(grid_features).features),
                                                   batch['input_sparse_sdf_idx_size'].to(
                                                       device),
                                                   dim=0)
            )
        elif config['vcdec_p_dim']:
            vc = vcdec(grid_features.features, query=input_sparse_xyz)
        else:
            vc = vcdec(grid_features.features)
        vc /= torch.repeat_interleave(batch['input_gs'].to(
            device), batch['input_sparse_sdf_idx_size'].to(device), dim=0).unsqueeze(1)

        if config['vcdec_vc_tanh']:
            vc *= config['vcdec_vc_tanh_scale']

        vc += input_sparse_xyz

        gt_clouds = batch['gt_cloud_glob'].to(device)
        l = 0
        r = 0
        loss = 0
        reg = 0
        for j in range(len(batch['input_sparse_sdf_idx_size'])):
            r += batch['input_sparse_sdf_idx_size'][j]
            loss += voroloss(gt_clouds[j].unsqueeze(0),
                             vc[l:r].unsqueeze(0)).sum()
            reg += config['reg_w'] * \
                torch.sqrt(((vc[l:r] - input_sparse_xyz[l:r])**2).sum(1)).max()
            l += batch['input_sparse_sdf_idx_size'][j]
        loss /= len(batch['input_sparse_sdf_idx_size'])
        reg /= len(batch['input_sparse_sdf_idx_size'])

        floss = loss + reg
        with torch.no_grad():
            if torch.isnan(floss):
                print('Loss is NaN! Stopping without updating the net...')
                exit()

        optimizer.zero_grad()
        floss.backward()
        optimizer.step()

        batch_time.update(time() - end)
        VL.update(loss.item(), batch['input_gs'].shape[0])
        R.update(reg.item(), batch['input_gs'].shape[0])

        # if (i + 1) % (config['num_workers'] // 2) == 0:
        line = 'Epoch: [{0}][{1}/{2}]'.format(epoch + 1,
                                              i + 1, len(train_iterator))
        line += '\tData {data_time.val:.3f} ({data_time.avg:.3f})'.format(
            data_time=data_time)
        line += '\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
            batch_time=batch_time)
        line += '\tVoroloss {VL.val:.3f} ({VL.avg:.3f})'.format(VL=VL)
        line += '\tReg {R.val:.3f} ({R.avg:.3f})'.format(R=R)
        tqdm.write(line)

        end = time()


def evaluation_step():
    pass


if __name__ == '__main__':
    parser = define_options_parser()
    args = parser.parse_args()
    with io.open(args.config, 'r') as stream:
        config = yaml.load(stream, yaml.Loader)
    config['reg_w'] = args.reg_w
    config['n_epochs'] = args.n_epochs
    config['lr'] = args.lr
    config['beta2'] = args.beta2
    config['resume'] = True if args.resume else False
    config['resume_optimizer'] = True if args.resume_optimizer else False
    print('Config file loaded.')

    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    input_pc_transform = ComposeCloudTransformation(**config)
    sample_transform = ComposeSampleTransformation(**config)
    train_dataset = ABCDataset(
        config['tr_obj_dir'], config['tr_surf_dir'], config['tr_sdf_dir'], None, val=False,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform
    )
    eval_dataset = ABCDataset(
        config['val_obj_dir'], config['val_surf_dir'], config['val_sdf_dir'], None, val=True,
        input_type=config['input_type'],
        grid_size=config['grid_size'], v_start=config['v_start'], v_end=config['v_end'],
        return_gt_mesh=True,
        n_input_pc=config['n_input_pc'], input_pc_transform=input_pc_transform,
        knn_input=config['knn_input'], local_input=config['local_input'], pooling_radius=config['pooling_radius'],
        truncate_sdf=config['truncate_sdf'], truncation_cell_dist=config['truncation_cell_dist'],
        sparse_sdf=config['sparse_sdf'],
        sample_grid_points=config['sample_grid_points'], sample_grid_size=config['sample_grid_size'],
        sample_grid_add_noise=config['sample_grid_add_noise'], sample_grid_add_noise_scale=config['sample_grid_add_noise_scale'],
        sample_gt_pc=config['sample_gt_pc'], n_gt_pc=config['n_gt_pc'],
        sample_transform=sample_transform,
    )
    print('Datasets init: done.')

    train_iterator = DataLoader(
        train_dataset, collate_fn=abc_collate_fn,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], drop_last=True
    )
    eval_iterator = DataLoader(
        eval_dataset, collate_fn=abc_collate_fn,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], drop_last=False
    )
    print('Iterator init: done.')

    encoder = LocalSDFFE_ME(
        config['in_dim'], config['enc_feat_dim']).to(device)
    if config['vcdec_in_glob']:
        maxpooler = ME.MinkowskiGlobalMaxPooling().to(device)
        glob_encoder = GlobFeatEnc(
            config['enc_feat_dim'][-1], [config['enc_feat_dim'][-1], config['enc_feat_dim'][-1]]).to(device)
    vcdec = VCDec(config['vcdec_p_dim'], config['enc_feat_dim'][-1], config['vcdec_feat_dim'], in_glob=config['vcdec_in_glob'],
                  vc_per_query=config['vcdec_vc_per_query'], vc_tanh=config['vcdec_vc_tanh'],
                  pe=config['vcdec_pe'], pe_feat_dim=config['vcdec_pe_feat_dim'],
                  film=config['vcdec_film'], film_std=config['vcdec_film_std']).to(device)
    print('Model init: done.')
    print('Total number of parameters in encoder: {}'.format(
        cnt_params(encoder.parameters())))
    if config['vcdec_in_glob']:
        print('Total number of parameters in global encoder: {}'.format(
            cnt_params(glob_encoder.parameters())))
    print('Total number of parameters in sdfnet: {}'.format(
        cnt_params(vcdec.parameters())))

    voroloss = Voroloss_opt().to(device)

    optimizer = AdamW([{'params': encoder.parameters()}, {'params': vcdec.parameters()}],
                      lr=config['lr'], weight_decay=config['wd'],
                      betas=(config['beta1'], config['beta2']), amsgrad=config['amsgrad'])
    print('Optimizer init: done.')

    if not config['resume']:
        cur_epoch = 0
        cur_iter = 0
    else:
        path2checkpoint = os.path.join(
            config['path2save'], config['model_name'])
        checkpoint = torch.load(path2checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state'])
        vcdec.load_state_dict(checkpoint['vcdec_state'])
        if config['vcdec_in_glob']:
            glob_encoder.load_state_dict(checkpoint['glob_encoder_state'])
        cur_epoch = checkpoint['voroloss_epoch']
        if config['resume_optimizer']:
            optimizer.load_state_dict(checkpoint['voroloss_optimizer_state'])
        del (checkpoint)
        print('Model {} loaded.'.format(path2checkpoint))

    for epoch in tqdm(range(cur_epoch, config['n_epochs']), initial=cur_epoch, total=config['n_epochs']):
        training_step()
        save_model({
            'encoder_state': encoder.state_dict(),
            'vcdec_state': vcdec.state_dict(),
            'glob_encoder_state': glob_encoder.state_dict() if config['vcdec_in_glob'] else None,
            'voroloss_epoch': epoch + 1,
            'voroloss_optimizer_state': optimizer.state_dict()
        }, os.path.join(config['path2save'], config['model_name']))
        tqdm.write('Model saved to: {}'.format(
            os.path.join(config['path2save'], config['model_name'])))

        if (epoch + 1) % 5 == 0:
            evaluation_step()
