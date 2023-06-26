"""
Search each step
Step 1 : Load Dataset
Step 2: Load Model
Step 3 : Load Rendering
Step 4 : Ray Generation
Step 5: Ray Colorization
"""

import _init_env

import yaml
import torch
import datetime
import argparse

import dataset
import model_nerf
import model_neus
import model_nerfies
# import model_hypernerf as hypernerf


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nerf')
parser.add_argument('--data', default='llff')


def get_configs(_data, _model_architecture):
    setting = yaml.safe_load(open(f'./configs/model_{_model_architecture}.yml'))
    embedding_cfg = setting['embed']
    dataset_cfg = setting['data'][_data]
    model_cfg = setting['model']
    rendering_cfg = setting['render']
    log_cfg = setting['log']
    run_cfg = setting['run']
    return embedding_cfg, dataset_cfg, model_cfg, rendering_cfg, log_cfg, run_cfg


def nerf_pipeline():
    global model_config
    global data

    # Step 1 : Load Dataset
    if data == 'llff':
        dset = dataset.LLFFDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num'], 'N_rots': rendering_config['N_rots'], 'zrate': rendering_config['zrate']}))
    else:
        dset = dataset.SyntheticDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Model
    model_config = dict(model_config, **rendering_config)
    # model_config['embed_cfg'] = embedding_config
    models, params, embedder_ray, embedder_view = model_nerf.get_model(**model_config)

    model_nerf.run(model_config, rendering_config, dataset_config, dset, params, models, embedder_ray, embedder_view)


def neus_pipeline():
    # Step 1 : Load Dataset
    dset = dataset.NeusDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Renderer and Model
    nerf_outside, sdf_network, deviation_network, color_network, params_to_train = model_neus.get_model(model_config)

    renderer = model_neus.NeuSRenderer(nerf_outside, sdf_network, deviation_network, color_network, **rendering_config)

    model_neus.run(model_config, rendering_config, dataset_config, log_config, params_to_train, renderer, dset)


def nerfies_pipeline():
    # Step 1 : Load Dataset
    dset = dataset.NerfiesDataSet.get_dataset(dataset_config, model_config)

    # Step 2: Load Model
    nerfies = model_nerfies.get_model(model_config, rendering_config, run_config, dset)

    model_nerfies.run(nerfies, dset, run_config, rendering_config, model_config)
    print()


def hypernerf_pipeline():
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    # model_architecture = args.model
    # data = args.data
    model_architecture = 'nerfies'
    data = 'broom'

    embedding_config, dataset_config, model_config, rendering_config, log_config, run_config = get_configs(data, model_architecture)
    # dataset_config, model_config, rendering_config, log_config, run_config = get_configs(data, model_architecture)
    # model_architecture = 'nerf'
    # data = 'synthetic'
    # # data = 'llff'
    # nerf_pipeline()

    # model_architecture = 'neus'
    # data = 'thin_structure'
    # neus_pipeline(args.model, args.data)

    # hypernerf_pipeline(args.model, args.data)
    nerfies_pipeline()

