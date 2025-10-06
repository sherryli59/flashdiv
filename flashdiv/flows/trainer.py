
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from einops import repeat, rearrange

from flashdiv.flows.pathgrad import reverse_kl_pathgrad_loss

class FlowTrainer(LightningModule):
    SUPPORTED_OBJECTIVES = {"flow_matching", "forward_kl", "pathgrad", "era"}

    def __init__(
        self,
        flow_model,
        learning_rate=1e-3,
        permute=False,
        sigma=0,
        objective: str = "flow_matching",
        lam_div: float = 3e-4, lam_v: float = 1e-3,
        base_distribution: Optional[object] = None,
        target_distribution: Optional[object] = None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.permute = permute
        self.sigma = sigma  # Standard deviation for noise
        self.lam_div = lam_div
        self.lam_v = lam_v
        if objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"Unsupported objective '{objective}'. Supported: {sorted(self.SUPPORTED_OBJECTIVES)}"
            )
        self.objective = objective
        self.base_distribution = base_distribution
        self.target_distribution = target_distribution
        self._current_weight: Optional[torch.Tensor] = None
        self._current_target_logprob: Optional[torch.Tensor] = None
        self.save_hyperparameters(ignore=["flow_model", "base_distribution", "target_distribution"])

    def permute_batch(self, batch):

        batchperm = torch.zeros_like(batch)
        perms = rearrange(
            [torch.randperm(batch.shape[1]) for _ in range(batch.shape[0])],
            'b p -> b p').flatten()

        flattened_range = repeat(
            torch.arange(batch.shape[0]),
            'b -> b p ',
            p = batch.shape[1]
            ).flatten()

        flattened_parts = repeat(
            torch.arange(batch.shape[1]),
            'p -> b p ',
            b = batch.shape[0]
            ).flatten()

        batchperm[flattened_range, flattened_parts] = batch[flattened_range, perms]
        return batchperm.detach()

    def forward(self, x, t):
        return self.flow_model(x, t)

    def training_step(self, batch, batch_idx):
        base, target = self._prepare_batch(batch)
        loss, metrics = self._apply_objective(base, target, stage="train")
        self._log_metrics(metrics, stage="train")
        return loss

    # def on_after_backward(self):
    #     # Access gradient after backward
    #     if hasattr(self, "_last_xs") and self._last_xs.grad is not None:
    #         grad_mean = self._last_xs.grad.abs().mean().item()
    #         print(f"[Gradient check] ∂loss/∂xs.mean(): {grad_mean:.4e}")
    #     else:
    #         print("[Gradient check] No gradient on xs!")

    def validation_step(self, batch, batch_idx):
        base, target = self._prepare_batch(batch)
        loss, metrics = self._apply_objective(base, target, stage="val")
        self._log_metrics(metrics, stage="val")
        return loss

    def on_after_backward(self):
        if not self.training:
            return
        grads = [p.grad for p in self.flow_model.parameters() if p.grad is not None]
        device = self.flow_model.parameters().__next__().device
        if not grads:
            zero = torch.tensor(0.0, device=device)
            self.log("grad_norm", zero, on_step=True, on_epoch=False, prog_bar=False)
            self.log("grad_norm_max", zero, on_step=True, on_epoch=False, prog_bar=False)
            return
        norms = torch.stack([g.detach().norm() for g in grads])
        self.log("grad_norm", norms.norm(), on_step=True, on_epoch=False, prog_bar=False)
        self.log("grad_norm_max", norms.max(), on_step=True, on_epoch=False, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------
    # Objective helpers
    # ------------------------------------------------------------------

    def _prepare_batch(self, batch):
        self._current_weight = None
        self._current_target_logprob = None
        if not isinstance(batch, (list, tuple)):
            raise TypeError(f"Expected batch as tuple/list, got {type(batch)!r}")

        base, target, *extras = batch
        if extras:
            extra = extras[0]
            if isinstance(extra, dict):
                logprob = extra.get("logprob")
                weight = extra.get("weight")
                if logprob is not None:
                    self._current_target_logprob = logprob
                if weight is not None:
                    self._current_weight = weight
            else:
                if self.objective == "era":
                    self._current_target_logprob = extra
                else:
                    self._current_weight = extra

            if len(extras) > 1:
                for item in extras[1:]:
                    if isinstance(item, dict):
                        if "logprob" in item:
                            self._current_target_logprob = item["logprob"]
                        if "weight" in item:
                            self._current_weight = item["weight"]
                    else:
                        if self._current_weight is None:
                            self._current_weight = item
        if self.permute:
            target = self.permute_batch(target)
        return base, target

    def _apply_objective(self, base, target, stage: str):
        if self.objective == "flow_matching":
            return self._flow_matching_loss(base, target, stage)
        if self.objective == "forward_kl":
            return self._forward_kl_loss(target, stage)
        if self.objective == "era":
            return self._era_loss(target, stage)
        if self.objective == "pathgrad":
            return self._pathgrad_loss(base, stage)
        
        raise RuntimeError(f"Unknown objective '{self.objective}'")

    def _flow_matching_loss(self, base, target, stage: str):
        t = torch.rand(base.shape[0], device=base.device)
        tr = t.view(-1, 1, 1)
        noise = self.sigma * torch.randn_like(base) if self.sigma else 0.0
        xt = base * (1 - tr) + target * tr + noise
        v = target - base

        weight = self._current_weight
        if weight is not None:
            weight = weight.to(device=base.device, dtype=base.dtype)
            if weight.dim() != 1:
                weight = weight.view(weight.size(0), -1).mean(dim=1)
            if stage == "val":
                weight = weight / torch.clamp(weight.sum(), min=1e-6)

        vt = self.flow_model(xt, t)
        per_sample = ((v - vt) ** 2).flatten(1).mean(dim=1)
        if weight is not None:
            loss = (weight * per_sample).mean()
            v_squared = (weight * (v ** 2).flatten(1).mean(dim=1)).mean()
        else:
            loss = per_sample.mean()
            v_squared = (v ** 2).flatten(1).mean(dim=1).mean()

        v_squared_norm = torch.clamp(v_squared, min=1e-12)
        metric_name = f"{stage}_loss"
        metrics = {metric_name: loss / v_squared_norm}
        return loss, metrics

    def _forward_kl_loss(self, target, stage: str):
        target = target.requires_grad_(True)
        if self.lam_div > 0:
            logprob, int_div2, int_v2 = self.flow_model.estimate_logprob(target, return_aux=True,
                base_dist=self.base_distribution, div_method='hutch', div_samples=4,differentiable=True)
        else:
            logprob = self.flow_model.estimate_logprob(target, return_aux=False,
                base_dist=self.base_distribution, div_method='hutch', div_samples=4,differentiable=True)
        reg = self.lam_div * int_div2.mean() + self.lam_v * int_v2.mean() if self.lam_div > 0 else 0.0
        mean_logprob = logprob.mean()
        loss = -mean_logprob + reg
        metrics = {f"{stage}_loss": loss}
        metrics[f"{stage}_logprob"] = mean_logprob.detach()
        if self.target_distribution is not None:
            with torch.no_grad():
                target_logprob = self._log_prob(self.target_distribution, target)
                metrics[f"{stage}_kl_est"] = (
                    target_logprob.mean() - mean_logprob
                ).detach()
        return loss, metrics

    def _pathgrad_loss(self, base, stage: str):
        t_grid = torch.linspace(0., 1., steps=10, device=base.device)
        loss, metrics = reverse_kl_pathgrad_loss(self.flow_model, self.target_distribution,base, t_grid)
        return loss, metrics
    
    def _era_loss(self, target, stage: str):
        perm = torch.randperm(target.shape[0], device=target.device)
        if self.lam_div > 0:
            logprob_model, int_div2, int_v2 = self.flow_model.estimate_logprob(target, return_aux=True,
                base_dist=self.base_distribution, differentiable=True)
        else:
            logprob_model = self.flow_model.estimate_logprob(
                target,
                base_dist=self.base_distribution,
                differentiable=True,
            )
        reg = self.lam_div * int_div2.mean() + self.lam_v * int_v2.mean() if self.lam_div > 0 else 0.0
        logprob_model_prime = logprob_model[perm]
        if self.target_distribution is None:
            raise RuntimeError(
                "ERA objective requires a target distribution or precomputed log-probabilities."
            )
        with torch.no_grad():
            logprob_target = self._log_prob(self.target_distribution, target).detach().flatten()
            logprob_target_prime = logprob_target[perm].flatten()
        dlog_model = logprob_model - logprob_model_prime
        dlog_target = logprob_target - logprob_target_prime
        logp_target_pref = torch.nn.functional.logsigmoid(dlog_target) #log p_target(y>y')
        logp_target_pref_reverse = torch.nn.functional.logsigmoid(-dlog_target) #log p_target(y'> y)
        p_target_ref = torch.exp(logp_target_pref) # p_target(y>y')
        logp_model_pref = torch.nn.functional.logsigmoid(dlog_model) #log p_model(y>y')
        logp_model_pref_reverse = torch.nn.functional.logsigmoid(-dlog_model) #log p_model(y'> y)
        loss_pair = p_target_ref * (logp_target_pref - logp_model_pref) + (1 - p_target_ref) * (logp_target_pref_reverse - logp_model_pref_reverse)
        loss_pair = loss_pair.mean()
        loss = loss_pair + reg
        metrics = {f"{stage}_era_loss": loss_pair.detach(),
                   f"{stage}_reg": reg.detach() if self.lam_div > 0 else 0.0}
        if stage == "val":
            metrics[f"{stage}_loss"] = loss_pair.detach()
        return loss, metrics


    def _log_prob(self, distribution, samples):
        if distribution is None:
            raise RuntimeError(
                f"Objective '{self.objective}' requires a distribution with log_prob"
            )
        log_prob_fn = getattr(distribution, "log_prob", None)
        if log_prob_fn is None:
            raise AttributeError(f"Distribution {distribution!r} does not implement log_prob")
        return log_prob_fn(samples)

    def _log_metrics(self, metrics, stage: str):
        for name, value in metrics.items():
            track_loss = name.endswith("_loss")
            if stage == "train":
                self.log(
                    f"{name}_step",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=track_loss,
                )
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
            else:
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=track_loss,
                )

# same as above but we use the shortest Torus path to do flow matching.
class FlowTrainerTorus(FlowTrainer):
    def __init__(
        self,
        flow_model,
        learning_rate=1e-3,
        permute=False,
        sigma=0,
        boxlength=None,
        objective: str = "flow_matching",
        base_distribution: Optional[object] = None,
        target_distribution: Optional[object] = None,
    ):
        box = 100.0 if boxlength is None else float(boxlength)
        if objective != "flow_matching":
            raise ValueError("FlowTrainerTorus currently supports only the flow_matching objective")
        self.boxlength = box
        super().__init__(
            flow_model,
            learning_rate=learning_rate,
            permute=permute,
            sigma=sigma,
            objective=objective,
            base_distribution=base_distribution,
            target_distribution=target_distribution,
        )
        self.save_hyperparameters({"boxlength": self.boxlength})

    def _flow_matching_loss(self, base, target, stage: str):
        weight = self._current_weight
        if weight is None:
            weight = torch.ones(base.shape[0], device=base.device, dtype=base.dtype)
        else:
            weight = weight.to(device=base.device, dtype=base.dtype)
            if weight.dim() != 1:
                weight = weight.view(weight.size(0), -1).mean(dim=1)

        if stage == "val":
            weight = weight / torch.clamp(weight.sum(), min=1e-6)

        t = torch.rand(base.shape[0], device=base.device)
        vtorus = target - base
        to_subtract = (
            (torch.abs(vtorus) > 0.5 * self.boxlength) * torch.sign(vtorus) * self.boxlength
        )
        vtorus = vtorus - to_subtract

        noise = self.sigma * torch.randn_like(base) if self.sigma else 0.0
        tr = t.view(-1, 1, 1)
        xt = base + tr * vtorus + noise
        xt = xt % self.boxlength
        vt = self.flow_model(xt, t)

        if self.lam_div > 0:
            vt, div = self.flow_model.forward_and_divergence(xt, t)
            reg = self.lam_div * (div ** 2).mean()
        else:
            vt = self.flow_model(xt, t)
            reg = 0.0
        if self.lam_v > 0:
            reg = reg + self.lam_v * (vt ** 2).mean()
        per_sample = ((vtorus - vt) ** 2).flatten(1).mean(1)
        mse_loss = (weight * per_sample).mean()
        denom = (weight * (vtorus ** 2).flatten(1).mean(1)).mean()
        scaled_loss = mse_loss / torch.clamp(denom, min=1e-12)
        loss = mse_loss + reg
        metrics = {f"{stage}_loss": scaled_loss,
                    f"{stage}_mse": mse_loss.detach(),
                    f"{stage}_reg": reg.detach() if self.lam_div > 0 else 0.0}
        return loss, metrics
    
  
