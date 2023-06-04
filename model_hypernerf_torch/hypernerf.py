import numpy as np
from typing import Dict, Any

import torch.nn as nn

from model_hypernerf_torch import embed
from model_hypernerf_torch import hypernerf_modules as modules
from model_hypernerf_torch import hypernerf_utils as model_utils
from model_hypernerf_torch.rendering import filter_sigma


def get_model(model_cfg, embed_cfg):
    model = _load_hypernerf(
        in_feature= model_cfg['in_feature'],
        trunk_layer_num=model_cfg['nerf_trunk_depth'],
        trunk_hidden_dim=model_cfg['nerf_trunk_width'],
        rgb_layer_num=model_cfg['nerf_rgb_branch_depth'],
        rgb_hidden_dim=model_cfg['nerf_rgb_branch_width'],
        rgb_channels=model_cfg['nerf_rgb_channels'],
        alpha_layer_num=model_cfg['nerf_alpha_depth'],
        alpha_hidden_dim=model_cfg['nerf_alapha_width'],
        alpha_channels=model_cfg['nerf_alpha_channels'],
        use_fine_model=model_cfg['use_fine_model'],
        emb_cfg=embed_cfg,
        skips=model_cfg['nerf_skips'],
        hypersheet_cfg=model_cfg['hypersheet']
    )
    return model


def _load_hypernerf(in_feature, trunk_layer_num, trunk_hidden_dim, rgb_layer_num, rgb_hidden_dim, rgb_channels,
                    alpha_layer_num, alpha_hidden_dim, alpha_channels, use_fine_model, emb_cfg, hypersheet_cfg, skips):
    """Neural Randiance Field.

    Args:
    key: jnp.ndarray. Random number generator.
    batch_size: the evaluation batch size used for shape inference.
    embeddings_dict: a dictionary containing the embeddings for each metadata
      type.
    near: the near plane of the scene.
    far: the far plane of the scene.

    Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
    """
    # TODO
    # Embedding
    if emb_cfg.use_nerf_embed:
        nerf_embed = embed.GLOEmbed(num_embeddings=emb_cfg.nerf_num_embeddings, num_dims=emb_cfg.nerf_num_dims)
    if emb_cfg.use_warp:
        warp_embed = embed.GLOEmbed(num_embeddings=emb_cfg.warp_num_embeddings, num_dims=emb_cfg.warp_num_dims)

    if emb_cfg.hyper_slice_method == 'axis_aligned_plane':
        hyper_embed = embed.GLOEmbed(num_embeddings=emb_cfg.hyper_num_embeddings, num_dims=emb_cfg.hyper_num_dims)
    elif emb_cfg.hyper_slice_method == 'bendy_sheet':
        if not emb_cfg.hyper_use_warp_embed:
            hyper_embed = embed.GLOEmbed(num_embeddings=emb_cfg.hyper_num_embeddings, num_dims=emb_cfg.hyper_num_dims)
        hyper_sheet_mlp = modules.HyperSheetMLP(
            in_channels=hypersheet_cfg['in_channels'],
            out_channels=hypersheet_cfg['out_channels'],
            min_deg=hypersheet_cfg['min_deg'],
            max_deg=hypersheet_cfg['max_deg'],
            depth=hypersheet_cfg['depth'],
            width=hypersheet_cfg['width'],
            skips=hypersheet_cfg['skips'],
            use_residual=hypersheet_cfg['use_residual'],
        )

    if emb_cfg.use_warp:
        emb_cfg.warp_field = emb_cfg.warp_field_cls()

    # Coarse model
    nerf_mlps = {
        'coarse': modules.NeRFMLP(in_feature, trunk_layer_num, trunk_hidden_dim,
                                  rgb_layer_num, rgb_hidden_dim, rgb_channels,
                                  alpha_layer_num, alpha_hidden_dim, alpha_channels,
                                  skipts=skips)
    }
    # Fine model
    if use_fine_model > 0:
        nerf_mlps['fine'] = modules.NeRFMLP(in_feature, trunk_layer_num, trunk_hidden_dim,
                                            rgb_layer_num, rgb_hidden_dim, rgb_channels,
                                            alpha_layer_num, alpha_hidden_dim, alpha_channels,
                                            skipts=skips)

    model = HyperNeRF(in_feature, nerf_mlps)
    return model


class HyperNeRF(nn.Module):
    def __init__(self, in_feature, nerf_mlps):
        super(HyperNeRF, self).__init__()
        self.norm_layer = modules.get_norm_layer(self.norm_type, in_feature)
        self.nerf_mlps = nerf_mlps

    @property
    def num_nerf_embeds(self):
        return max(self.embeddings_dict[self.nerf_embed_key]) + 1

    @property
    def num_warp_embeds(self):
        return max(self.embeddings_dict[self.warp_embed_key]) + 1

    @property
    def num_hyper_embeds(self):
        return max(self.embeddings_dict[self.hyper_embed_key]) + 1

    @property
    def nerf_embeds(self):
        return np.array(self.embeddings_dict[self.nerf_embed_key], np.uint32)

    @property
    def warp_embeds(self):
        return np.array(self.embeddings_dict[self.warp_embed_key], np.uint32)

    @property
    def hyper_embeds(self):
        return np.array(self.embeddings_dict[self.hyper_embed_key], np.uint32)

    @property
    def has_hyper(self):
        """Whether the model uses a separate hyper embedding."""
        return self.hyper_slice_method != 'none'

    @property
    def has_hyper_embed(self):
        """Whether the model uses a separate hyper embedding."""
        # If the warp field outputs the hyper coordinates then there is no separate
        # hyper embedding.
        return self.has_hyper

    @property
    def has_embeds(self):
        return self.has_hyper_embed or self.use_warp or self.use_nerf_embed

    def encode_hyper_embed(self, metadata):
        if self.hyper_slice_method == 'axis_aligned_plane':
            if self.hyper_use_warp_embed:
                return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)
            else:
                return self._encode_embed(metadata[self.hyper_embed_key], self.hyper_embed)
        elif self.hyper_slice_method == 'bendy_sheet':
            # The bendy sheet shares the metadata of the warp.
            if self.hyper_use_warp_embed:
                return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)
            else:
                return self._encode_embed(metadata[self.hyper_embed_key], self.hyper_embed)
        else:
            raise RuntimeError(f'Unknown hyper slice method {self.hyper_slice_method}.')

    def encode_nerf_embed(self, metadata):
        return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

    def encode_warp_embed(self, metadata):
        return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)

    def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
        """Create the condition inputs for the NeRF template."""
        alpha_conditions = []
        rgb_conditions = []

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_feat = model_utils.posenc(
              viewdirs,
              min_deg=self.viewdir_min_deg,
              max_deg=self.viewdir_max_deg,
              use_identity=self.use_posenc_identity)
            rgb_conditions.append(viewdirs_feat)

        if self.use_nerf_embed:
            if metadata_encoded:
                nerf_embed = metadata['encoded_nerf']
            else:
                nerf_embed = metadata[self.nerf_embed_key]
                nerf_embed = self.nerf_embed(nerf_embed)
            if self.use_alpha_condition:
                alpha_conditions.append(nerf_embed)
            if self.use_rgb_condition:
                rgb_conditions.append(nerf_embed)

        # The condition inputs have a shape of (B, C) now rather than (B, S, C)
        # since we assume all samples have the same condition input. We might want
        # to change this later.
        alpha_conditions = (
            np.concatenate(alpha_conditions, axis=-1)
            if alpha_conditions else None)
        rgb_conditions = (
            np.concatenate(rgb_conditions, axis=-1)
            if rgb_conditions else None)
        return alpha_conditions, rgb_conditions

    def query_template(self,
                     level,
                     points,
                     viewdirs,
                     metadata,
                     extra_params,
                     metadata_encoded=False):
        """Queries the NeRF template."""
        alpha_condition, rgb_condition = (
            self.get_condition_inputs(viewdirs, metadata, metadata_encoded))

        points_feat = model_utils.posenc(
            points[..., :3],
            min_deg=self.spatial_point_min_deg,
            max_deg=self.spatial_point_max_deg,
            use_identity=self.use_posenc_identity,
            alpha=extra_params['nerf_alpha'])
        # Encode hyper-points if present.
        if points.shape[-1] > 3:
          hyper_feats = model_utils.posenc(
              points[..., 3:],
              min_deg=self.hyper_point_min_deg,
              max_deg=self.hyper_point_max_deg,
              use_identity=False,
              alpha=extra_params['hyper_alpha'])
          points_feat = np.concatenate([points_feat, hyper_feats], axis=-1)

        raw = self.nerf_mlps[level](points_feat, alpha_condition, rgb_condition)
        raw = model_utils.noise_regularize(
            self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling)

        rgb = nn.sigmoid(raw['rgb'])
        sigma = self.sigma_activation(np.squeeze(raw['alpha'], axis=-1))

        return rgb, sigma

    # TODO
    # def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
    #                      return_warp_jacobian=False):
    #     warp_jacobian = None
    #     if self.use_warp and use_warp:
    #       warp_fn = jax.vmap(jax.vmap(self.warp_field, in_axes=(0, 0, None, None)),
    #                          in_axes=(0, 0, None, None))
    #       warp_out = warp_fn(points,
    #                          warp_embed,
    #                          extra_params,
    #                          return_warp_jacobian)
    #       if return_warp_jacobian:
    #         warp_jacobian = warp_out['jacobian']
    #       warped_points = warp_out['warped_points']
    #     else:
    #       warped_points = points
    #
    #     return warped_points, warp_jacobian

    def map_hyper_points(self, points, hyper_embed, extra_params,
                       hyper_point_override=None):
        """Maps input points to hyper points.

        Args:
          points: the input points.
          hyper_embed: the hyper embeddings.
          extra_params: extra params to pass to the slicing MLP if applicable.
          hyper_point_override: this may contain an override for the hyper points.
            Useful for rendering at specific hyper dimensions.

        Returns:
          An array of hyper points.
        """
        if hyper_point_override is not None:
          hyper_points = np.broadcast_to(
              hyper_point_override[:, None, :],
              (*points.shape[:-1], hyper_point_override.shape[-1]))
        elif self.hyper_slice_method == 'axis_aligned_plane':
          hyper_points = hyper_embed
        elif self.hyper_slice_method == 'bendy_sheet':
          hyper_points = self.hyper_sheet_mlp(
              points,
              hyper_embed,
              alpha=extra_params['hyper_sheet_alpha'])
        else:
          return None

        return hyper_points

    def map_points(self, points, warp_embed, hyper_embed, extra_params, use_warp=True, return_warp_jacobian=False,
                 hyper_point_override=None):
        """Map input points to warped spatial and hyper points.

        Args:
          points: the input points to warp.
          warp_embed: the warp embeddings.
          hyper_embed: the hyper embeddings.
          extra_params: extra parameters to pass to the warp field/hyper field.
          use_warp: whether to use the warp or not.
          return_warp_jacobian: whether to return the warp jacobian or not.
          hyper_point_override: this may contain an override for the hyper points.
            Useful for rendering at specific hyper dimensions.

        Returns:
          A tuple containing `(warped_points, warp_jacobian)`.
        """
        # Map input points to warped spatial and hyper points.
        spatial_points, warp_jacobian = self.map_spatial_points(
            points, warp_embed, extra_params, use_warp=use_warp,
            return_warp_jacobian=return_warp_jacobian)
        hyper_points = self.map_hyper_points(
            points, hyper_embed, extra_params,
            # Override hyper points if present in metadata dict.
            hyper_point_override=hyper_point_override)

        if hyper_points is not None:
          warped_points = np.concatenate([spatial_points, hyper_points], axis=-1)
        else:
          warped_points = spatial_points

        return warped_points, warp_jacobian

    def apply_warp(self, points, warp_embed, extra_params):
        warp_embed = self.warp_embed(warp_embed)
        return self.warp_field(points, warp_embed, extra_params)

    def render_samples(self,
                         level,
                         points,
                         z_vals,
                         directions,
                         viewdirs,
                         metadata,
                         extra_params,
                         use_warp=True,
                         metadata_encoded=False,
                         return_warp_jacobian=False,
                         use_sample_at_infinity=False,
                         render_opts=None):
        out = {'points': points}

        batch_shape = points.shape[:-1]
        # Create the warp embedding.
        if use_warp:
          if metadata_encoded:
            warp_embed = metadata['encoded_warp']
          else:
            warp_embed = metadata[self.warp_embed_key]
            warp_embed = self.warp_embed(warp_embed)
        else:
          warp_embed = None

        # Create the hyper embedding.
        if self.has_hyper_embed:
          if metadata_encoded:
            hyper_embed = metadata['encoded_hyper']
          elif self.hyper_use_warp_embed:
            hyper_embed = warp_embed
          else:
            hyper_embed = metadata[self.hyper_embed_key]
            hyper_embed = self.hyper_embed(hyper_embed)
        else:
          hyper_embed = None

        # Broadcast embeddings.
        if warp_embed is not None:
          warp_embed = np.broadcast_to(
              warp_embed[:, np.newaxis, :],
              shape=(*batch_shape, warp_embed.shape[-1]))
        if hyper_embed is not None:
          hyper_embed = np.broadcast_to(
              hyper_embed[:, np.newaxis, :],
              shape=(*batch_shape, hyper_embed.shape[-1]))

        # Map input points to warped spatial and hyper points.
        warped_points, warp_jacobian = self.map_points(
            points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
            return_warp_jacobian=return_warp_jacobian,
            # Override hyper points if present in metadata dict.
            hyper_point_override=metadata.get('hyper_point'))

        rgb, sigma = self.query_template(
            level,
            warped_points,
            viewdirs,
            metadata,
            extra_params=extra_params,
            metadata_encoded=metadata_encoded)

        # Filter densities based on rendering options.
        sigma = filter_sigma(points, sigma, render_opts)

        if warp_jacobian is not None:
          out['warp_jacobian'] = warp_jacobian
        out['warped_points'] = warped_points
        out.update(model_utils.volumetric_rendering(
            rgb,
            sigma,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sample_at_infinity=use_sample_at_infinity))

        # Add a map containing the returned points at the median depth.
        depth_indices = model_utils.compute_depth_index(out['weights'])
        med_points = np.take_along_axis(
            # Unsqueeze axes: sample axis, coords.
            warped_points, depth_indices[..., None, None], axis=-2)
        out['med_points'] = med_points

        return out

    def __call__(
          self,
          rays_dict: Dict[str, Any],
          extra_params: Dict[str, Any],
          metadata_encoded=False,
          use_warp=True,
          return_points=False,
          return_weights=False,
          return_warp_jacobian=False,
          near=None,
          far=None,
          use_sample_at_infinity=None,
          render_opts=None,
          deterministic=False,
        ):
        """Nerf Model.

        Args:
          rays_dict: a dictionary containing the ray information. Contains:
            'origins': the ray origins.
            'directions': unit vectors which are the ray directions.
            'viewdirs': (optional) unit vectors which are viewing directions.
            'metadata': a dictionary of metadata indices e.g., for warping.
          extra_params: parameters for the warp e.g., alpha.
          metadata_encoded: if True, assume the metadata is already encoded.
          use_warp: if True use the warp field (if also enabled in the model).
          return_points: if True return the points (and warped points if
            applicable).
          return_weights: if True return the density weights.
          return_warp_jacobian: if True computes and returns the warp Jacobians.
          near: if not None override the default near value.
          far: if not None override the default far value.
          use_sample_at_infinity: override for `self.use_sample_at_infinity`.
          render_opts: an optional dictionary of render options.
          deterministic: whether evaluation should be deterministic.

        Returns:
          ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
        """
        use_warp = self.use_warp and use_warp
        # Extract viewdirs from the ray array
        origins = rays_dict['origins']
        directions = rays_dict['directions']
        metadata = rays_dict['metadata']
        if 'viewdirs' in rays_dict:
          viewdirs = rays_dict['viewdirs']
        else:  # viewdirs are normalized rays_d
          viewdirs = directions

        if near is None:
          near = self.near
        if far is None:
          far = self.far
        if use_sample_at_infinity is None:
          use_sample_at_infinity = self.use_sample_at_infinity

        # Evaluate coarse samples.
        z_vals, points = model_utils.sample_along_rays(
            self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
            near, far, self.use_stratified_sampling,
            self.use_linear_disparity)
        coarse_ret = self.render_samples(
            'coarse',
            points,
            z_vals,
            directions,
            viewdirs,
            metadata,
            extra_params,
            use_warp=use_warp,
            metadata_encoded=metadata_encoded,
            return_warp_jacobian=return_warp_jacobian,
            use_sample_at_infinity=self.use_sample_at_infinity)
        out = {'coarse': coarse_ret}

        # Evaluate fine samples.
        if self.num_fine_samples > 0:
          z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
          z_vals, points = model_utils.sample_pdf(
              self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
              origins, directions, z_vals, self.num_fine_samples,
              self.use_stratified_sampling)
          out['fine'] = self.render_samples(
              'fine',
              points,
              z_vals,
              directions,
              viewdirs,
              metadata,
              extra_params,
              use_warp=use_warp,
              metadata_encoded=metadata_encoded,
              return_warp_jacobian=return_warp_jacobian,
              use_sample_at_infinity=use_sample_at_infinity,
              render_opts=render_opts)

        if not return_weights:
          del out['coarse']['weights']
          del out['fine']['weights']

        if not return_points:
          del out['coarse']['points']
          del out['coarse']['warped_points']
          del out['fine']['points']
          del out['fine']['warped_points']

        return out

