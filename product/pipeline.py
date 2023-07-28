import dataset
import model_nerf
import model_neus
import model_nerfies


def get_pipeline(model_type: str):
    if model_type.lower() == 'nerf':
        return nerf_pipeline
    elif model_type.lower() == 'neus':
        return neus_pipeline
    elif model_type.lower() == 'nerfies':
        return nerfies_pipeline
    else:
        raise AssertionError(f"Cannot Support Model type of {model_type}")


def nerf_pipeline(dataset_config, rendering_config, model_config, log_config, run_config):

    # Step 1 : Load Dataset
    if dataset_config['data_type'] == 'llff':
        dset = dataset.LLFFDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num'], 'N_rots': rendering_config['N_rots'], 'zrate': rendering_config['zrate']}))
    else:
        dset = dataset.SyntheticDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Model
    model_config = dict(model_config, **rendering_config)
    # model_config['embed_cfg'] = embedding_config
    models, params, embedder_ray, embedder_view = model_nerf.get_model(**model_config)

    model_nerf.run(model_config, rendering_config, dataset_config, dset, params, models, embedder_ray, embedder_view)


def neus_pipeline(dataset_config, rendering_config, model_config, log_config, run_config):
    # Step 1 : Load Dataset
    dset = dataset.NeusDataset(**dict(dataset_config, **{'render_pose_num': rendering_config['render_pose_num']}))

    # Step 2: Load Renderer and Model
    nerf_outside, sdf_network, deviation_network, color_network, params_to_train = model_neus.get_model(model_config)

    renderer = model_neus.NeuSRenderer(nerf_outside, sdf_network, deviation_network, color_network, **rendering_config)

    model_neus.run(model_config, rendering_config, dataset_config, log_config, params_to_train, renderer, dset)


def nerfies_pipeline(dataset_config, rendering_config, model_config, log_config, run_config):
    # Step 1 : Load Dataset
    dset = dataset.NerfiesDataSet.get_dataset(dataset_config, model_config)

    # Step 2: Load Model
    nerfies = model_nerfies.get_model(model_config, rendering_config, run_config, dset)

    model_nerfies.run(nerfies, dset, run_config, rendering_config, model_config, log_config)
