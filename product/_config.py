import yaml


def get_configs(_data, _model_type):
    setting = yaml.safe_load(open(f'./configs/model_{_model_type}.yml'))
    embedding_cfg = setting['embed']
    if _data not in setting['data'].keys():
        _data = setting['data']['default']
    dataset_cfg = setting['data'][_data]
    model_cfg = setting['model']
    rendering_cfg = setting['render']
    log_cfg = setting['log']
    run_cfg = setting['run']
    return embedding_cfg, dataset_cfg, model_cfg, rendering_cfg, log_cfg, run_cfg


def put_config(_config, config_name):
    try:
        with open(f'./configs/{config_name}.png', 'w') as f:
            yaml.dump(_config, f)
    except:
        raise AssertionError("Cannot write config")


