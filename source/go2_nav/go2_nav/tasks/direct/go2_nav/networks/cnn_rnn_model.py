# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
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
    """CNN -> RNN sequential model.

    This model uses CNN encoders for 2D observation groups.
    The CNN latent is concatenated with the 1D observation groups and then passed to an RNN.
    The RNN latent is then passed to an MLP head.
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

        # RNN for 1D observations + cnn latent.
        self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

        # Register CNN encoders.
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining RNN-encoded 1D and CNN-encoded 2D observation groups."""
        # Process 2D observations with CNN
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

        # Get 1D observations
        latent_1d = super().get_latent(obs)

        # Concatenate and pass to RNN
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)

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
        raise NotImplementedError("ONNX export not implemented for CNNRNNSeqModel.")

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
        return self.rnn_hidden_dim



class _TorchGRUCNNRNNSeqModel(nn.Module):
    """Exportable GRU CNN-RNN-Seq model for JIT."""

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


class _TorchLSTMCNNRNNSeqModel(nn.Module):
    """Exportable LSTM CNN-RNN-Seq model for JIT."""

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
        self.cell_state[:] = 0.0  # type: ignoreSeq