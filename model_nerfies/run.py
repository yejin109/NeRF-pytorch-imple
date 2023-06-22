from model_nerfies.nerfies import get_model

# 1. Ray Generation
# 2. Encoding
# 3. Ray Sampling
# 4. Ray Colorization
# 5. Rendering


def train(model):
    pass


def inference():
    pass


def run(model_cfg, render_cfg):
        # Model config
        # use_fine_samples, coarse_args, fine_args,
        # use_warp, warp_field_args, use_warp_jacobian,
        # use_appearance_metadata, appearance_encoder_args,
        # use_camera_metadata, camera_encoder_args,
        # use_trunk_condition, use_alpha_condition):
    model = get_model(model_cfg, render_cfg)

    return model
