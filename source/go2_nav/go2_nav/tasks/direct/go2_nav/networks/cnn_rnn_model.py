# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, RNN, HiddenState
from rsl_rl.utils import unpad_trajectories


class CNNRNNModel(MLPModel):
    """CNN + RNN model.

    This model uses CNN encoders for 2D observation groups and an RNN over 1D observation groups.
    The RNN latent is concatenated with the CNN latents before passing to an MLP head.
    """

    is_recurrent: bool = True

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
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        # Resolve observation groups and dimensions for CNN construction.
        self._get_obs_dim(obs, obs_groups, obs_set)

        # Create or validate CNN encoders.
        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of observation groups.")
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    **cnn_cfg[obs_group],
                )

        # Compute latent dimension of the CNNs.
        self.cnn_latent_dim = 0
        for cnn in cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
            self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore

        self.rnn_hidden_dim = rnn_hidden_dim

        # Initialize the parent MLP model.
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

        # RNN for 1D observations.
        self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

        # Register CNN encoders.
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining RNN-encoded 1D and CNN-encoded 2D observation groups."""
        latent_1d = super().get_latent(obs)
        latent_1d = self.rnn(latent_1d, masks, hidden_state).squeeze(0)

        latent_cnn_list = []
        for obs_group in self.obs_groups_2d:
            obs_2d = obs[obs_group]
            if masks is not None:
                obs_2d = unpad_trajectories(obs_2d, masks)
                time_len, batch_len = obs_2d.shape[0], obs_2d.shape[1]
                obs_2d = obs_2d.reshape(time_len * batch_len, *obs_2d.shape[2:])
                latent_cnn = self.cnns[obs_group](obs_2d)
                latent_cnn = latent_cnn.reshape(time_len, batch_len, -1)
            else:
                latent_cnn = self.cnns[obs_group](obs_2d)
            latent_cnn_list.append(latent_cnn)
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        return torch.cat([latent_1d, latent_cnn], dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMCNNRNNModel(self)
        if isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUCNNRNNModel(self)
        raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNRNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute 1D observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. Use RNNModel if this is intentional.")

        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d

        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.rnn_hidden_dim + self.cnn_latent_dim


class _TorchGRUCNNRNNModel(nn.Module):
    """Exportable GRU CNN-RNN model for JIT."""

    def __init__(self, model: CNNRNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        latent_1d, h = self.rnn(latent_1d.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        latent_1d = latent_1d.squeeze(0)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMCNNRNNModel(nn.Module):
    """Exportable LSTM CNN-RNN model for JIT."""

    def __init__(self, model: CNNRNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        latent_1d, (h, c) = self.rnn(latent_1d.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        latent_1d = latent_1d.squeeze(0)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxCNNRNNModel(nn.Module):
    """Exportable CNN-RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: CNNRNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(self, obs_1d: torch.Tensor, *obs_2d_and_state: torch.Tensor):
        if self.rnn_type == "lstm":
            *obs_2d, h_in, c_in = obs_2d_and_state
            latent_1d = self.obs_normalizer(obs_1d)
            latent_1d, (h, c) = self.rnn(latent_1d.unsqueeze(0), (h_in, c_in))
        else:
            *obs_2d, h_in = obs_2d_and_state
            latent_1d = self.obs_normalizer(obs_1d)
            latent_1d, h = self.rnn(latent_1d.unsqueeze(0), h_in)
            c = None

        latent_1d = latent_1d.squeeze(0)
        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        out = self.mlp(latent)
        out = self.deterministic_output(out)
        if self.rnn_type == "lstm":
            return out, h, c
        return out, h

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (dummy_1d, *dummy_2d, h_in, c_in)
        return (dummy_1d, *dummy_2d, h_in)

    @property
    def input_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["obs", *self.obs_groups_2d, "h_in", "c_in"]
        return ["obs", *self.obs_groups_2d, "h_in"]

    @property
    def output_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]

        
        
class CNNRNNSeqModel(MLPModel):
    """
    """

    is_recurrent: bool = True

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
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        # Resolve observation groups and dimensions for CNN construction.
        self._get_obs_dim(obs, obs_groups, obs_set)

        # Create or validate CNN encoders.
        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of observation groups.")
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx],
                    **cnn_cfg[obs_group],
                )

        # Compute latent dimension of the CNNs.
        self.cnn_latent_dim = 0
        for cnn in cnns.values():
            if cnn.output_channels is not None:
                raise ValueError("The output of the CNN must be flattened before passing it to the MLP.")
            self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore

        self.rnn_hidden_dim = rnn_hidden_dim

        # Initialize the parent MLP model.
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

        # RNN consumes the FUSED (1D + CNN) per-timestep latent, not the 1D latent alone.
        self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

        # Register CNN encoders.
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Fuse CNN-encoded 2D and raw 1D observation groups, then run the RNN over the fused sequence."""
        # Raw (un-encoded) 1D latent, same shape convention as the base MLPModel: (B, D) for inference,
        # (T, B, D) for a masked/padded training rollout.
        latent_1d = super().get_latent(obs)

        latent_cnn_list = []
        for obs_group in self.obs_groups_2d:
            obs_2d = obs[obs_group]
            if masks is not None:
                # NOTE: unlike CNNRNNModel, we do NOT unpad_trajectories here before the CNN forward.
                # latent_1d above is still in its full padded (T, B, D) shape, and the two streams must
                # share that shape to be concatenated before the RNN does its own internal masking/packing.
                time_len, batch_len = obs_2d.shape[0], obs_2d.shape[1]
                obs_2d_flat = obs_2d.reshape(time_len * batch_len, *obs_2d.shape[2:])
                latent_cnn = self.cnns[obs_group](obs_2d_flat)
                latent_cnn = latent_cnn.reshape(time_len, batch_len, -1)
            else:
                latent_cnn = self.cnns[obs_group](obs_2d)
            latent_cnn_list.append(latent_cnn)
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        # Fuse before the RNN: this is the key difference from CNNRNNModel.
        latent_fused = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent = self.rnn(latent_fused, masks, hidden_state).squeeze(0)

        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMCNNRNNSeqModel(self)
        if isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUCNNRNNSeqModel(self)
        raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNRNNSeqModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute 1D observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. Use RNNModel if this is intentional.")

        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d

        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head.

        Unlike CNNRNNModel, the CNN latent is consumed BY the RNN (fused in before it runs),
        not concatenated after it -- so the MLP head only sees the RNN's hidden output.
        """
        return self.rnn_hidden_dim


class _TorchGRUCNNRNNSeqModel(nn.Module):
    """Exportable GRU CNN-RNN-New model for JIT."""

    def __init__(self, model: CNNRNNSeqModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        # Fuse before the RNN.
        latent_fused = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent, h = self.rnn(latent_fused.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        latent = latent.squeeze(0)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMCNNRNNSeqModel(nn.Module):
    """Exportable LSTM CNN-RNN-New model for JIT."""

    def __init__(self, model: CNNRNNSeqModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        # Fuse before the RNN.
        latent_fused = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent, (h, c) = self.rnn(latent_fused.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        latent = latent.squeeze(0)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxCNNRNNSeqModel(nn.Module):
    """Exportable CNN-RNN-New model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: CNNRNNSeqModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim + model.cnn_latent_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(self, obs_1d: torch.Tensor, *obs_2d_and_state: torch.Tensor):
        if self.rnn_type == "lstm":
            *obs_2d, h_in, c_in = obs_2d_and_state
        else:
            *obs_2d, h_in = obs_2d_and_state
            c_in = None

        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent_cnn = torch.cat(latent_cnn_list, dim=-1)

        # Fuse before the RNN.
        latent_fused = torch.cat([latent_1d, latent_cnn], dim=-1)
        if self.rnn_type == "lstm":
            latent, (h, c) = self.rnn(latent_fused.unsqueeze(0), (h_in, c_in))
        else:
            latent, h = self.rnn(latent_fused.unsqueeze(0), h_in)
            c = None

        latent = latent.squeeze(0)
        out = self.mlp(latent)
        out = self.deterministic_output(out)
        if self.rnn_type == "lstm":
            return out, h, c
        return out, h

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (dummy_1d, *dummy_2d, h_in, c_in)
        return (dummy_1d, *dummy_2d, h_in)

    @property
    def input_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["obs", *self.obs_groups_2d, "h_in", "c_in"]
        return ["obs", *self.obs_groups_2d, "h_in"]

    @property
    def output_names(self) -> list[str]:
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
    
    
    
    
    

HiddenState = torch.Tensor | tuple[torch.Tensor, ...] | None

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


def _make_mlp(in_dim: int, hidden_dims: tuple[int, ...] | list[int], activation: str) -> nn.Sequential:
    """MLP whose last layer is also followed by an activation (it feeds a FiLM head)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), _ACTIVATIONS[activation]()]
        prev = h
    return nn.Sequential(*layers)

class ConvGRUCell(nn.Module):
    """GRU cell whose gates are convolutions; hidden state is a spatial map (C, h, w)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=pad)
        self.cand = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], dim=1))).chunk(2, dim=1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * n


class SpatialMemoryRNN(nn.Module):
    """Spatial recurrent core.

    Per step:
      1. Encode the raw scan with a small, shape-preserving conv trunk (stride 1, same padding) so its
         spatial layout AND its metric resolution stay known and unchanged.
      2. Geometrically warp the encoded scan onto the global (state) grid with `grid_sample`, using the
         robot's current (x, y, yaw): this is an explicit, non-learned placement, not something the
         network has to infer. Cells the scan doesn't cover receive zero.
      3. Concatenate the warped scan content with the (dx, dy, r) geometry field and run `coord_net`
         on BOTH together, so content and geometry interact locally and spatially, not just through a
         single global gate.
      4. Feed the result into the ConvGRU.

    This replaces the earlier FiLM-only design, where the scan only modulated the geometry branch
    globally (one gamma/beta per channel), which let the network collapse to a content-independent
    radial function of (dx, dy, r) alone -- visible as concentric rings in the predicted map.
    """

    def __init__(
        self,
        obs_dim_1d: int,
        map_shape: tuple[int, int],
        map_resolution: float,
        map_origin: tuple[float, float],
        state_stride: int,
        scan_shape: tuple[int, int],
        scan_resolution: float,
        scan_channels: int,
        scan_center: tuple[float, float] | None,
        scan_yaw_aligned: bool,
        scan_feat_channels: int,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        fusion_dims: tuple[int, ...] | list[int],
        activation: str,
        odom_indices: tuple[int, int, int, int],  # (x, y, sin_yaw, cos_yaw)
        coord_scale: float | None,
    ) -> None:
        super().__init__()
        height, width = map_shape
        self.state_h = math.ceil(height / state_stride)
        self.state_w = math.ceil(width / state_stride)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.x_idx, self.y_idx, self.sin_idx, self.cos_idx = odom_indices
        self.coord_scale = coord_scale if coord_scale is not None else 0.5 * max(height, width) * map_resolution

        self.scan_h, self.scan_w = scan_shape
        self.scan_resolution = scan_resolution
        self.scan_yaw_aligned = scan_yaw_aligned
        # Scan-grid index of the robot's own position within the local scan window.
        # Default: geometric center. Override if the sensor window is offset (e.g. forward-facing).
        sci, scj = scan_center if scan_center is not None else ((self.scan_h - 1) / 2.0, (self.scan_w - 1) / 2.0)
        self.scan_center_i = sci
        self.scan_center_j = scj

        # Shape-preserving scan encoder: stride 1, same padding -> output stays (scan_h, scan_w),
        # so the metric resolution (scan_resolution) used for the warp below stays exact.
        self.scan_encoder = nn.Sequential(
            nn.Conv2d(scan_channels, scan_feat_channels, 3, padding=1),
            _ACTIVATIONS[activation](),
            nn.Conv2d(scan_feat_channels, scan_feat_channels, 3, padding=1),
        )

        # Small conditioning MLP from odometry only, used for a light global FiLM on top of the
        # (now content-aware) geometry branch -- kept only for extra context (e.g. velocity), not as
        # the sole content path anymore.
        self.fusion = nn.Sequential(
            nn.Linear(obs_dim_1d, fusion_dims[0]), _ACTIVATIONS[activation](),
            nn.Linear(fusion_dims[0], fusion_dims[-1]), _ACTIVATIONS[activation](),
        )
        self.film = nn.Linear(fusion_dims[-1], 2 * input_channels)

        # coord_net now takes geometry (3 ch) + warped scan content (scan_feat_channels).
        self.coord_net = nn.Sequential(
            nn.Conv2d(3 + scan_feat_channels, input_channels, 3, padding=1),
            _ACTIVATIONS[activation](),
            nn.Conv2d(input_channels, input_channels, 3, padding=1),
        )
        self.act = _ACTIVATIONS[activation]()

        cells = []
        for i in range(num_layers):
            cells.append(ConvGRUCell(input_channels if i == 0 else hidden_channels, hidden_channels))
        self.cells = nn.ModuleList(cells)

        # World coordinates of state-cell centers (fixed, precomputed once).
        ox, oy = map_origin
        ci = (torch.arange(self.state_h) * state_stride + (state_stride - 1) / 2.0 + 0.5) * map_resolution + ox
        cj = (torch.arange(self.state_w) * state_stride + (state_stride - 1) / 2.0 + 0.5) * map_resolution + oy
        self.register_buffer("grid_x", ci.view(1, -1, 1).expand(1, self.state_h, self.state_w).clone(), persistent=False)
        self.register_buffer("grid_y", cj.view(1, 1, -1).expand(1, self.state_h, self.state_w).clone(), persistent=False)

        self.hidden_state: torch.Tensor | None = None

    # ---------------------------------------------------------------- geometric warp
    def _warp_scan(self, scan_feat: torch.Tensor, x: torch.Tensor, y: torch.Tensor, sin_yaw: torch.Tensor, cos_yaw: torch.Tensor) -> torch.Tensor:
        """Place scan_feat (B, C, scan_h, scan_w) onto the (state_h, state_w) grid via grid_sample.

        For every output cell, computes its world offset from the robot, converts to the scan's local
        frame (rotating by -yaw if the scan is yaw-aligned to the robot body), and samples scan_feat at
        that location. Cells outside the scan's coverage sample as zero (padding_mode="zeros").
        """
        # World-frame offset of every state cell from the robot, in meters.
        owx = self.grid_x - x.view(-1, 1, 1)  # (B, state_h, state_w)
        owy = self.grid_y - y.view(-1, 1, 1)

        if self.scan_yaw_aligned:
            # Rotate world offset into the robot's body frame (inverse of the robot's world rotation).
            local_x = owx * cos_yaw.view(-1, 1, 1) + owy * sin_yaw.view(-1, 1, 1)
            local_y = -owx * sin_yaw.view(-1, 1, 1) + owy * cos_yaw.view(-1, 1, 1)
        else:
            local_x, local_y = owx, owy

        # Convert metric body-frame offset to fractional scan-grid indices.
        si = local_x / self.scan_resolution + self.scan_center_i
        sj = local_y / self.scan_resolution + self.scan_center_j

        # grid_sample expects normalized coords in [-1, 1], last dim = (x -> width/scan_w, y -> height/scan_h).
        norm_x = 2.0 * sj / max(self.scan_w - 1, 1) - 1.0
        norm_y = 2.0 * si / max(self.scan_h - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, state_h, state_w, 2)

        warped = F.grid_sample(scan_feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped  # (B, C, state_h, state_w)

    # ---------------------------------------------------------------- core step
    def forward_step(
        self,
        height_data: torch.Tensor,
        latent_1d: torch.Tensor,
        raw_1d: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = raw_1d[..., self.x_idx]
        y = raw_1d[..., self.y_idx]
        sin_yaw = raw_1d[..., self.sin_idx]
        cos_yaw = raw_1d[..., self.cos_idx]

        scan_feat = self.scan_encoder(height_data)                      # (B, C, scan_h, scan_w)
        warped = self._warp_scan(scan_feat, x, y, sin_yaw, cos_yaw)      # (B, C, state_h, state_w)

        owx = self.grid_x - x.view(-1, 1, 1)
        owy = self.grid_y - y.view(-1, 1, 1)
        dx = owx / self.coord_scale
        dy = owy / self.coord_scale
        r = torch.sqrt(dx * dx + dy * dy + 1e-6)
        coords = torch.stack([dx, dy, r], dim=1)

        coord_in = torch.cat([coords, warped], dim=1)
        g = self.coord_net(coord_in)

        f = self.fusion(latent_1d)
        gamma, beta = self.film(f).chunk(2, dim=-1)
        out = self.act(g * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1))

        new_h: list[torch.Tensor] = []
        for i, cell in enumerate(self.cells):
            h_i = cell(out, hidden[i])
            new_h.append(h_i)
            out = h_i
        return out, torch.stack(new_h, dim=0)

    # ---------------------------------------------------------------- hidden state management
    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_channels, self.state_h, self.state_w, device=device, dtype=dtype
        )

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if hidden_state is not None:
            self.hidden_state = hidden_state  # type: ignore
            return
        if self.hidden_state is None:
            return
        if dones is None:
            self.hidden_state = torch.zeros_like(self.hidden_state)
        else:
            new_hidden = self.hidden_state.clone()
            new_hidden[:, dones.bool()] = 0.0
            self.hidden_state = new_hidden

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()


class MapDecoder(nn.Module):
    """Conv decoder: (..., C, h, w) -> upsample to (H, W) -> 2 output channels -> flatten."""

    def __init__(self, in_channels: int, map_shape: tuple[int, int], decoder_channels: int, activation: str) -> None:
        super().__init__()
        self.map_h, self.map_w = map_shape
        act = _ACTIVATIONS[activation]
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, 3, padding=1), act(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1), act(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1), act(),
            nn.Conv2d(decoder_channels, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = list(x.shape[:-3])
        x = x.reshape([-1] + list(x.shape[-3:]))
        x = self.pre(x)
        if x.shape[-2] != self.map_h or x.shape[-1] != self.map_w:
            x = F.interpolate(x, size=[self.map_h, self.map_w], mode="bilinear", align_corners=False)
        x = self.post(x)
        return x.reshape(lead + [-1])


class CNNConvGRUMapModel(MLPModel):
    """Content-aware ConvGRU map model: scan is warped onto the global frame geometrically, then
    fused with (dx, dy, r) before the conv/ConvGRU stack, instead of only modulating it via FiLM.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (64, 64),  # odom fusion MLP only now
        activation: str = "elu",
        obs_normalization: bool = False,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 32,
        rnn_num_layers: int = 1,
        map_shape: tuple[int, int] = (250, 250),
        map_resolution: float = 0.2,
        map_origin: tuple[float, float] | None = None,
        state_stride: int = 2,
        scan_resolution: float = 0.2,
        scan_center: tuple[float, float] | None = None,
        scan_yaw_aligned: bool = True,  # set False if the scan is world/spawn-axis-aligned, not robot-yaw-aligned
        scan_feat_channels: int = 16,
        input_channels: int = 32,
        decoder_channels: int = 32,
        odom_indices: tuple[int, int, int, int] = (0, 1, 5, 6),  # x, y, sin_yaw, cos_yaw
        coord_scale: float | None = None,
    ) -> None:
        if rnn_type != "gru":
            raise ValueError("CNNConvGRUMapModel only supports rnn_type='gru' (ConvGRU).")
        if output_dim != 2 * map_shape[0] * map_shape[1]:
            raise ValueError(f"output_dim must be 2 * H * W = {2 * map_shape[0] * map_shape[1]}, got {output_dim}.")

        self._get_obs_dim(obs, obs_groups, obs_set)
        if len(self.obs_groups_2d) != 1:
            raise ValueError("CNNConvGRUMapModel expects exactly one 2D observation group (the height scan).")
        scan_group = self.obs_groups_2d[0]
        scan_channels = self.obs_channels_2d[0]
        scan_shape = tuple(self.obs_dims_2d[0])
        self.scan_group = scan_group

        if map_origin is None:
            map_origin = (-0.5 * map_shape[0] * map_resolution, -0.5 * map_shape[1] * map_resolution)

        self.rnn_hidden_dim = rnn_hidden_dim
        self.map_shape = tuple(map_shape)

        # output_dim=1 placeholder: the dense head is replaced by MapDecoder below.
        super().__init__(obs, obs_groups, obs_set, 1, (8,), activation, obs_normalization, None)

        self.rnn = SpatialMemoryRNN(
            obs_dim_1d=self.obs_dim,
            map_shape=self.map_shape,
            map_resolution=map_resolution,
            map_origin=map_origin,
            state_stride=state_stride,
            scan_shape=scan_shape,
            scan_resolution=scan_resolution,
            scan_channels=scan_channels,
            scan_center=scan_center,
            scan_yaw_aligned=scan_yaw_aligned,
            scan_feat_channels=scan_feat_channels,
            input_channels=input_channels,
            hidden_channels=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            fusion_dims=hidden_dims,
            activation=activation,
            odom_indices=odom_indices,
            coord_scale=coord_scale,
        )

        self.mlp = MapDecoder(rnn_hidden_dim, self.map_shape, decoder_channels, activation)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_1d = super().get_latent(obs)
        raw_1d = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        scan = obs[self.scan_group]

        if masks is not None:
            if hidden_state is None:
                raise ValueError("A hidden state is required when masks are provided (batch mode).")
            h = hidden_state
            outs = []
            for t in range(latent_1d.shape[0]):
                out, h = self.rnn.forward_step(scan[t], latent_1d[t], raw_1d[t], h)
                outs.append(out)
            return unpad_trajectories(torch.stack(outs, dim=0), masks)

        batch = latent_1d.shape[0]
        hs = self.rnn.hidden_state
        if hs is None or hs.shape[1] != batch:
            hs = self.rnn.init_hidden(batch, latent_1d.device, latent_1d.dtype)
        elif hs.is_inference():
            hs = hs.clone()
        out, new_hs = self.rnn.forward_step(scan, latent_1d, raw_1d, hs)
        self.rnn.hidden_state = new_hs.detach()
        return out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")
        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided.")
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        return self.rnn_hidden_dim
    
    
    
class MapDenoiser(nn.Module):
    """Pure spatial refinement: takes the already-fused (h_hat, confidence) grid, built externally
    by an exact Kalman elevation-map update, and outputs a denoised/completed version. No odometry,
    no scan, no recurrence, no geometry inside this module -- all of that is handled upstream.
    """
    def __init__(self, in_channels: int = 2, base_channels: int = 32, activation: str = "elu"):
        super().__init__()
        act = _ACTIVATIONS[activation]

        def block(cin, cout, stride=1):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), act())

        # Encoder (downsample twice)
        self.enc1 = block(in_channels, base_channels)                 # H, W
        self.enc2 = block(base_channels, base_channels * 2, stride=2)  # H/2
        self.enc3 = block(base_channels * 2, base_channels * 4, stride=2)  # H/4

        # Bottleneck
        self.bottleneck = block(base_channels * 4, base_channels * 4)

        # Decoder (upsample twice, with skip connections)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = block(base_channels * 4, base_channels * 2)  # concat skip from enc2
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = block(base_channels * 2, base_channels)      # concat skip from enc1

        self.out_conv = nn.Conv2d(base_channels, 2, 1)  # height, confidence_logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        up2 = F.interpolate(self.up2(b), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(self.up1(d2), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))
        output = self.out_conv(d1)
        return F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)