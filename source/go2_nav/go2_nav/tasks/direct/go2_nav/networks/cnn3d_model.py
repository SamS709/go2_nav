# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
import torch
from torch import nn as nn

from typing import Any
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel

from rsl_rl.utils import get_param, resolve_nn_activation


class CNN3D(nn.Sequential):
    """A 3D Convolutional Neural Network, adapted from rsl_rl's 2D CNN.

    This network is a sequence of 3D convolutional layers, with optional normalization, activation, and pooling.
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        input_channels: int,
        output_channels: list[int],
        kernel_size: int | list,
        stride: int | list = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ):
        """Initialize the 3D CNN.

        Args:
            input_dim: Depth, height, and width of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each convolutional layer.
            kernel_size: List of kernel sizes for each layer.
            stride: List of strides for each layer.
            dilation: List of dilations for each convolutional layer.
            padding: Padding type.
            norm: Normalization type.
            activation: Activation function.
            max_pool: Whether to apply max pooling.
            global_pool: Global pooling type.
            flatten: Whether to flatten the output.
        """
        super().__init__()

        activation_function = resolve_nn_activation(activation)
        layers = []
        last_channels = input_channels
        last_dim = input_dim

        for idx in range(len(output_channels)):
            # Get parameters for the current layer
            k = get_param(kernel_size, idx)
            s = get_param(stride, idx)
            d = get_param(dilation, idx)
            
            # Compute padding
            padding_mode = padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros"
            p = _compute_padding_3d(last_dim, k, s, d) if padding_mode != "zeros" else (0, 0, 0)

            # Append convolutional layer
            layers.append(
                nn.Conv3d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding_mode,
                )
            )

            # Append normalization layer if specified
            n = get_param(norm, idx)
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm3d(output_channels[idx]))
            elif n == "layer":
                norm_input_dim = _compute_output_dim_3d(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_input_dim[0], norm_input_dim[1], norm_input_dim[2]]))
            else:
                raise ValueError(f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'.")

            layers.append(activation_function)

            # Apply max pooling if specified
            if get_param(max_pool, idx):
                layers.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))

            # Update last channels and dimensions
            last_channels = output_channels[idx]
            last_dim = _compute_output_dim_3d(last_dim, k, s, d, p, is_max_pool=get_param(max_pool, idx))

        # Apply global pooling if specified
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool3d((1, 1, 1)))
            last_dim = (1, 1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
            last_dim = (1, 1, 1)
        else:
            raise ValueError(f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'.")

        if flatten:
            layers.append(nn.Flatten(start_dim=1))

        self._output_channels = last_channels if not flatten else None
        self._output_dim = last_dim if not flatten else last_channels * last_dim[0] * last_dim[1] * last_dim[2]

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_dim(self) -> int:
        return self._output_dim

def _compute_padding_3d(input_dhw: tuple[int, int, int], kernel: int, stride: int, dilation: int) -> tuple[int, int, int]:
    """Compute the optimal padding for a 3D convolution."""
    d = math.ceil((stride * math.floor(input_dhw[0] / stride) - input_dhw[0] - stride + dilation * (kernel - 1) + 1) / 2)
    h = math.ceil((stride * math.floor(input_dhw[1] / stride) - input_dhw[1] - stride + dilation * (kernel - 1) + 1) / 2)
    w = math.ceil((stride * math.floor(input_dhw[2] / stride) - input_dhw[2] - stride + dilation * (kernel - 1) + 1) / 2)
    return (d, h, w)

def _compute_output_dim_3d(
    input_dhw: tuple[int, int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int, int],
    is_max_pool: bool = False,
) -> tuple[int, int, int]:
    """Compute the output dimension of a 3D convolutional layer."""
    # Ensure kernel, stride, and dilation are tuples
    if isinstance(kernel, int):
        kernel = (kernel, kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    d = math.floor((input_dhw[0] + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    h = math.floor((input_dhw[1] + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1)
    w = math.floor((input_dhw[2] + 2 * padding[2] - dilation[2] * (kernel[2] - 1) - 1) / stride[2] + 1)

    if is_max_pool:
        # Assuming MaxPool3d with kernel=3, stride=2, padding=1
        d = math.floor((d + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        h = math.floor((h + 2 * 1 - (3 - 1) - 1) / 2 + 1)
        w = math.floor((w + 2 * 1 - (3 - 1) - 1) / 2 + 1)

    return (d, h, w)



class CNN3DModel(MLPModel):
    """A model that uses a 3D CNN for perception and an MLP head."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        # Resolve observation groups and dimensions for CNN construction.
        self._get_obs_dim(obs, obs_groups, obs_set)

        if cnn_cfg is None:
            raise ValueError("3D CNN configuration must be provided.")

        # If a single CNN config is provided, create a dictionary for all 3D observation groups.
        if not all(isinstance(v, dict) for v in cnn_cfg.values()):
            cnn_cfg = {group: cnn_cfg for group in self.obs_groups_3d}
        # Check if the number of CNN configurations matches the number of 3D observation groups.
        if len(cnn_cfg) != len(self.obs_groups_3d):
            raise ValueError(
                f"The number of CNN configurations ({len(cnn_cfg)}) must match the number of 3D observation groups"
                f" ({len(self.obs_groups_3d)})."
            )

        # Create CNN encoders in a temporary dictionary.
        cnns = {}
        cnn_latent_dim = 0
        for idx, obs_group in enumerate(self.obs_groups_3d):
            cnn = CNN3D(
                input_dim=self.obs_dims_3d[idx],
                input_channels=self.obs_channels_3d[idx],
                **cnn_cfg[obs_group],
            )
            cnns[obs_group] = cnn
            cnn_latent_dim += cnn.output_dim

        # Set the CNN latent dimension before calling the parent constructor.
        self.cnn_latent_dim = cnn_latent_dim

        # Initialize the parent MLP model.
        # This will call `_get_latent_dim`, which now has access to `self.cnn_latent_dim`.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # Now, register the CNN encoders as a ModuleDict.
        self.cnns = nn.ModuleDict(cnns)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None) -> torch.Tensor:
        """Build the model latent by combining 1D and CNN-encoded 3D observation groups."""
        # Process 3D observations with CNN
        latent_cnn_list = []
        for obs_group in self.obs_groups_3d:
            obs_3d = obs[obs_group]
            latent_cnn = self.cnns[obs_group](obs_3d)
            latent_cnn_list.append(latent_cnn)
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        # Get 1D observations
        latent_1d = super().get_latent(obs)

        # Concatenate and return
        return torch.cat([latent_1d, latent_cnn], dim=-1)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimensions."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_3d = []
        obs_channels_3d = []
        obs_groups_3d = []

        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 5:  # B, C, D, H, W
                obs_groups_3d.append(obs_group)
                obs_channels_3d.append(obs[obs_group].shape[1])
                obs_dims_3d.append(obs[obs_group].shape[2:]) # (D, H, W)
            elif len(obs[obs_group].shape) == 2:  # B, F
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}. CNN3DModel only supports 5D (B,C,D,H,W) and 2D (B,F) tensors.")

        if not obs_groups_3d:
            raise ValueError("No 3D observations are provided. Use a different model if this is intentional.")

        self.obs_dims_3d = obs_dims_3d
        self.obs_channels_3d = obs_channels_3d
        self.obs_groups_3d = obs_groups_3d
        
        # This is for the parent class, which only knows about 1D obs
        self.obs_groups = {"1d": obs_groups_1d}
        self.obs_dim = obs_dim_1d

        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.cnn_latent_dim
