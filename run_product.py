import _init_env


import product
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model_type',
                    default='neus',
                    help="Model to be run")
parser.add_argument('--data',
                    default='custom')


if __name__ == '__main__':
    args = parser.parse_args()
    pipeline = product.get_pipeline(args.model_type)
    
    # NOTE: Embedding cfg는 Nerfies 이후로 사용하지 않음
    embedding_cfg, dataset_cfg, model_cfg, rendering_cfg, log_cfg, run_cfg = product.get_configs(args.data, args.model_type)
    
    # pipeline(dataset_cfg, rendering_cfg, model_cfg, log_cfg, run_cfg)
    print()
