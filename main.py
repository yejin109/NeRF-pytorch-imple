"""
Search each step
Step 1 : Load Dataset
Step 2: Load Model
Step 3 : Load Rendering
Step 4 : Ray Generation
Step 5: Ray Colorization
"""

import _init_env
import datetime


import yaml
import torch

import dataset
import model_nerf
import model_neus


def get_configs(_data, _model_architecture):
    embedding_cfg = setting['embed']
    dataset_cfg = setting['data'][data]
    model_cfg = setting['model'][model_architecture]
    rendering_cfg = setting['rendering'][model_architecture]
    log_cfg = setting['log']
    return embedding_cfg, dataset_cfg, model_cfg, rendering_cfg, log_cfg


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    setting = yaml.safe_load(open('./config.yml'))

    # TODO: argparse로 나중에 분리하기
    # model_architecture = 'nerf'
    # data = 'synthetic'
    # # data = 'llff'
    
    # embedding_config, dataset_config, model_config, rendering_config, log_config = get_configs(data, model_architecture)
    
    # # Step 1 : Load Dataset
    # if data == 'llff':
    #     dset = dataset.LLFFDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num'], 'N_rots': rendering_config['N_rots'], 'zrate': rendering_config['zrate']}))
    # else:
    #     dset = dataset.SyntheticDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))
    
    # # Step 2: Load Model
    # model_config = dict(model_config, **rendering_config)
    # model_config['embed_cfg'] = embedding_config
    # models, params, embedder_ray, embedder_view = model_nerf.get_model(**model_config)
    
    # model_nerf.run(model_config, rendering_config, dataset_config, dset, params, models, embedder_ray, embedder_view)

    model_architecture = 'neus'
    data = 'thin_structure'

    embedding_config, dataset_config, model_config, rendering_config, log_config = get_configs(data, model_architecture)

    # Step 1 : Load Dataset
    dset = dataset.NeusDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Renderer and Model
    nerf_outside, sdf_network, deviation_network, color_network, params_to_train = model_neus.get_model(model_config)

    renderer = model_neus.NeuSRenderer(nerf_outside, sdf_network, deviation_network, color_network, **rendering_config)

    model_neus.run(model_config, rendering_config, dataset_config, log_config, params_to_train, renderer, dset)

