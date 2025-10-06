from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class RealNVPCoupling(nn.Module):
    """Single affine coupling layer with a fixed binary mask."""

    def __init__(
        self,
        state_dim: int,
        mask: torch.Tensor,
        hidden_dim: int,
        num_hidden_layers: int,
        time_embed_dim: int,
        scale_clip: float,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if mask.ndim != 1 or mask.shape[0] != state_dim:
            raise ValueError("mask must be 1-D with length state_dim")
        mask_bool = mask.bool()
        if mask_bool.all() or (~mask_bool).all():
            raise ValueError("mask must select at least one and not all dimensions")
        self.register_buffer("mask", mask_bool, persistent=False)
        self.state_dim = state_dim
        self.scale_clip = float(scale_clip)

        cond_idx = mask_bool.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
        target_idx = (~mask_bool).nonzero(as_tuple=False).flatten().to(dtype=torch.long)
        self.register_buffer("cond_idx", cond_idx, persistent=False)
        self.register_buffer("target_idx", target_idx, persistent=False)
        cond_dim = cond_idx.numel()
        target_dim = target_idx.numel()

        layers: list[nn.Module] = []
        in_dim = cond_dim + time_embed_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2 * target_dim))
        self.net = nn.Sequential(*layers)
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling transformation and return the updated state and log-diagonal."""
        if x.shape[-1] != self.state_dim:
            raise ValueError("Input has wrong feature dimension")
        x_cond = x.index_select(dim=-1, index=self.cond_idx)
        h = torch.cat([x_cond, t_emb], dim=-1)
        shift_and_scale = self.net(h)
        shift, log_scale = shift_and_scale.split(self.target_dim, dim=-1)
        log_scale = torch.tanh(log_scale) * self.scale_clip
        scale = torch.exp(log_scale)

        y = x.clone()
        y_target = x.index_select(dim=-1, index=self.target_idx)
        y_target = y_target * scale + shift
        y[:, self.target_idx] = y_target

        log_diag = torch.zeros_like(x)
        log_diag[:, self.target_idx] = log_scale
        return y, log_diag

    def inverse(self, y: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the coupling layer and return the pre-image and log-diagonal of J_{f^{-1}}."""
        if y.shape[-1] != self.state_dim:
            raise ValueError("Input has wrong feature dimension")
        y_cond = y.index_select(dim=-1, index=self.cond_idx)
        h = torch.cat([y_cond, t_emb], dim=-1)
        shift_and_scale = self.net(h)
        shift, log_scale = shift_and_scale.split(self.target_dim, dim=-1)
        log_scale = torch.tanh(log_scale) * self.scale_clip
        scale = torch.exp(log_scale)

        x = y.clone()
        y_target = y.index_select(dim=-1, index=self.target_idx)
        x_target = (y_target - shift) / scale
        x[:, self.target_idx] = x_target

        inv_log_diag = torch.zeros_like(y)
        inv_log_diag[:, self.target_idx] = -log_scale
        return x, inv_log_diag


class RealNVP(nn.Module):
    """RealNVP-style coupling network compatible with FlowTrainer APIs."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_coupling_layers: int = 4,
        num_hidden_layers: int = 2,
        time_embed_dim: int | None = None,
        scale_clip: float = 2.0,
        output_difference: bool = True,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        if dim <= 1:
            raise ValueError("dim must be greater than 1 for RealNVP")
        if num_coupling_layers < 1:
            raise ValueError("At least one coupling layer is required")
        self.state_dim = dim
        self.hidden_dim = hidden_dim
        self.num_coupling_layers = num_coupling_layers
        self.time_embed_dim = time_embed_dim or hidden_dim
        self.scale_clip = float(scale_clip)
        self.output_difference = output_difference

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            activation(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            activation(),
        )

        base_mask = (torch.arange(dim) % 2 == 0)
        self.register_buffer("_base_mask", base_mask, persistent=False)

        couplings = []
        for layer_idx in range(num_coupling_layers):
            mask = base_mask if layer_idx % 2 == 0 else (~base_mask)
            couplings.append(
                RealNVPCoupling(
                    state_dim=dim,
                    mask=mask.to(torch.bool),
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    time_embed_dim=self.time_embed_dim,
                    scale_clip=self.scale_clip,
                    activation=activation,
                )
            )
        self.couplings = nn.ModuleList(couplings)
        self.time_embedding = self._concat_time_embedding

    def _concat_time_embedding(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_vec = t.reshape(-1, 1)
        t_emb = self.time_mlp(t_vec)
        return torch.cat([x, t_emb], dim=-1)

    def _transform(
        self,
        x_flat: torch.Tensor,
        t: torch.Tensor,
        *,
        track_log_diag: bool = False,
        differentiable: bool = False,
        reverse: bool = False,
        return_history: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            t_emb = self.time_mlp(t.reshape(-1, 1))
            y = x_flat
            log_diag_acc = torch.zeros_like(x_flat) if track_log_diag else None
            history: list[torch.Tensor] | None = [] if return_history else None
            couplings = reversed(self.couplings) if reverse else self.couplings
            for coupling in couplings:
                if reverse:
                    y, log_diag = coupling.inverse(y, t_emb)
                else:
                    y, log_diag = coupling(y, t_emb)
                if track_log_diag:
                    log_diag_acc = log_diag_acc + log_diag
                if history is not None:
                    history.append(y)
            return y, log_diag_acc, history

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, npart, feat_dim = x.shape
        x_flat = rearrange(x, "b part d -> b (part d)")
        y, _, _ = self._transform(
            x_flat,
            t,
            track_log_diag=False,
            differentiable=self.training,
            reverse=False,
            return_history=False,
        )
        v_flat = y - x_flat if self.output_difference else y
        return rearrange(v_flat, "b (part d) -> b part d", part=npart, d=feat_dim)

    def divergence(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        differentiable: bool = False,
    ) -> torch.Tensor:
        batch, npart, feat_dim = x.shape
        x_flat = rearrange(x, "b part d -> b (part d)")
        _, log_diag, _ = self._transform(
            x_flat,
            t,
            track_log_diag=True,
            differentiable=differentiable,
            reverse=False,
            return_history=False,
        )
        if log_diag is None:
            raise RuntimeError("log_diag should not be None when track_log_diag is True")
        diag_y = torch.exp(log_diag)
        div = (diag_y - 1.0).sum(dim=-1)
        return div

    def sample_logprob(
        self,
        x: torch.Tensor,
        logprob: torch.Tensor | None = None,
        times: torch.Tensor | None = None,
        reverse: bool = False,
        *,
        differentiable: bool = False,
        return_traj: bool = True,
        **kwargs,
    ):
        device = x.device
        dtype = x.dtype
        batch, npart, feat_dim = x.shape

        # Consume kwargs used by FlowNet interface but irrelevant here
        kwargs.pop("div_method", None)
        kwargs.pop("div_samples", None)

        if logprob is None:
            if reverse:
                logprob = torch.zeros(batch, device=device, dtype=dtype)
            else:
                raise ValueError("logprob must be provided when reverse=False for RealNVP sampling")
        else:
            logprob = logprob.to(device=device, dtype=dtype)
        logprob_initial = logprob.clone()

        t_vals = kwargs.pop("time", None)
        if t_vals is None:
            t_vals = torch.ones(batch, device=device, dtype=dtype)
        elif t_vals.ndim == 0:
            t_vals = t_vals.expand(batch).to(device=device, dtype=dtype)
        elif t_vals.shape[0] != batch:
            t_vals = t_vals.expand(batch).to(device=device, dtype=dtype)
        else:
            t_vals = t_vals.to(device=device, dtype=dtype)

        x_flat = rearrange(x, "b part d -> b (part d)")
        y_flat, log_diag, history = self._transform(
            x_flat,
            t_vals,
            track_log_diag=True,
            differentiable=differentiable,
            reverse=reverse,
            return_history=return_traj,
        )
        if log_diag is None:
            raise RuntimeError("RealNVP.transform did not return log-diagonal information")
        log_det = log_diag.sum(dim=-1)
        if reverse:
            logprob = logprob + log_det
        else:
            logprob = logprob - log_det

        y = rearrange(y_flat, "b (part d) -> b part d", part=npart, d=feat_dim)

        if not return_traj:
            return y, logprob

        traj_states = [rearrange(x_flat, "b (part d) -> b part d", part=npart, d=feat_dim)]
        if history is not None:
            for state in history:
                traj_states.append(rearrange(state, "b (part d) -> b part d", part=npart, d=feat_dim))
        else:
            traj_states.append(y)
        xs_traj = torch.stack(traj_states)
        lp_traj = torch.stack([logprob_initial, logprob])
        return y, logprob, xs_traj, lp_traj

    def estimate_logprob(
        self,
        x: torch.Tensor,
        base_dist,
        times: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("div_method", None)
        kwargs.pop("div_samples", None)
        differentiable = kwargs.pop("differentiable", False)
        z, log_det_inv = self.sample_logprob(
            x,
            logprob=None,
            times=times,
            reverse=True,
            differentiable=differentiable,
            return_traj=False,
            **kwargs,
        )
        base_logprob = base_dist.log_prob(z)
        return base_logprob + log_det_inv
