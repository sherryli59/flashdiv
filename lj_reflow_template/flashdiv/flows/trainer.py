
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from einops import repeat, rearrange, reduce
import torch.nn.functional as F
import time


class PathGrad(torch.autograd.Function):
    """Custom autograd shim to route gradients through ``x`` only.

    The forward pass returns ``l`` unchanged, while the backward pass supplies
    the gradient of ``l`` with respect to ``x`` and blocks gradients flowing
    directly to ``l``.
    """

    @staticmethod
    def forward(ctx, l, x):
        ctx.save_for_backward(l, x)
        return l

    @staticmethod
    def backward(ctx, g):
        l, x = ctx.saved_tensors
        grad_x = torch.autograd.grad(l, x, g, retain_graph=True)[0]
        return None, grad_x

class FlowTrainer(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3, permute=False, sigma = 0,
                 cfm_weight: float = 1.0, ml_weight: float = 0.0,
                 div_method: str = 'full_jacobian', div_samples: int = 1,
                 target_distribution=None, weight_var_weight: float = 0.0):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.permute = permute
        self.sigma = sigma  # Standard deviation for noise
        self.cfm_weight = cfm_weight
        self.ml_weight = ml_weight
        self.div_method = div_method
        self.div_samples = div_samples
        self.target_distribution = target_distribution
        self.weight_var_weight = weight_var_weight
        self.save_hyperparameters(ignore=["target_distribution"])

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
        if len(batch) == 3:
            base, target, weight = batch
        else:
            base, target = batch
            weight = torch.ones(base.shape[0], device=base.device)

        if self.permute:
            # permute the batch to avoid symmetry issues
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)
        v = target - base
        tr = t.view(-1, 1, 1)
        xt = base + tr * v + self.sigma * torch.randn_like(base)

        vt = self.flow_model(xt, t)

        per_sample_loss = F.mse_loss(vt, v, reduction='none').mean(dim=[1, 2])
        weight = weight.to(per_sample_loss.device)
        cfm_loss = (weight * per_sample_loss).sum() / weight.sum()

        ml_loss = torch.tensor(0.0, device=base.device)
        weight_var = torch.tensor(0.0, device=base.device)
        if self.ml_weight > 0:
            if self.target_distribution is None:
                raise ValueError("target_distribution must be provided for ML training")
            with torch.no_grad():
                target_score = self.target_distribution.grad_log_prob(target)
            x0, model_score, logdet = self.flow_model.integrate_augmented_adj(
                target,
                target_score,
                div_method=self.div_method,
                div_samples=self.div_samples,
            )
            prior_logp = -0.5 * (x0 ** 2).sum(dim=[1, 2])
            prior_score = -x0
            N = x0.shape[1] * x0.shape[2]
            grad_x0 = (model_score - prior_score) / N
            path_term = PathGrad.apply((x0 * grad_x0.detach()).sum(dim=[1, 2]), x0)
            ell = prior_logp + logdet
            ml_loss = -(ell.detach() + path_term).mean()

        if self.target_distribution is not None or self.weight_var_weight > 0:
            start = time.time()
            logprob0 = torch.zeros(base.shape[0], device=base.device)
            if self.weight_var_weight > 0:
                xs_traj, logq_traj = self.flow_model.sample_logprob(
                    base, logprob0, div_method=self.div_method,
                    div_samples=self.div_samples, boxlength=None,
                    differentiable=True,
                )
            else:
                with torch.no_grad():
                    xs_traj, logq_traj = self.flow_model.sample_logprob(
                        base, logprob0, div_method=self.div_method,
                        div_samples=self.div_samples, boxlength=None,
                        differentiable=False,
                    )
            xs = xs_traj[-1]
            log_q = logq_traj[-1]
            with torch.no_grad():
                if self.target_distribution is not None:
                    log_target = -self.target_distribution.potential(xs) / self.target_distribution.kT
                else:
                    log_target = torch.zeros_like(log_q)
            log_w = log_target - log_q
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = w / w.sum()
            ess = 1.0 / (w.pow(2).sum())
            ess_norm = ess / w.shape[0]
            weight_var = w.var(unbiased=False)
            self.log('train_ess', ess_norm, on_step=True, on_epoch=True)
            if self.weight_var_weight > 0:
                self.log('train_weight_var', weight_var, on_step=True, on_epoch=True)

        v_squared_norm = (v**2).mean()
        total_loss = (
            self.cfm_weight * cfm_loss
            + self.ml_weight * ml_loss
            + self.weight_var_weight * weight_var
        )
        self.log("train_loss", total_loss / v_squared_norm, on_step=True, on_epoch=True)
        if self.ml_weight > 0:
            self.log('train_ml', ml_loss, on_step=True, on_epoch=True)
        self.log('train_cfm', cfm_loss, on_step=True, on_epoch=True)
        return total_loss

    # def on_after_backward(self):
    #     # Access gradient after backward
    #     if hasattr(self, "_last_xs") and self._last_xs.grad is not None:
    #         grad_mean = self._last_xs.grad.abs().mean().item()
    #         print(f"[Gradient check] ∂loss/∂xs.mean(): {grad_mean:.4e}")
    #     else:
    #         print("[Gradient check] No gradient on xs!")

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            base, target, weight = batch
        else:
            base, target = batch
            weight = torch.ones(base.shape[0], device=base.device)

        if self.permute:
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)
        v = target - base
        tr = t.view(-1, 1, 1)
        xt = base + tr * v + self.sigma * torch.randn_like(base)

        vt = self.flow_model(xt, t)

        per_sample_loss = F.mse_loss(vt, v, reduction='none').mean(dim=[1, 2])
        weight = weight.to(per_sample_loss.device)
        cfm_loss = (weight * per_sample_loss).sum() / weight.sum()

        ml_loss = torch.tensor(0.0, device=base.device)
        weight_var = torch.tensor(0.0, device=base.device)
        if self.ml_weight > 0:
            with torch.no_grad():
                target_score = self.target_distribution.grad_log_prob(target)
                x0, model_score, logdet = self.flow_model.integrate_augmented_adj(
                    target,
                    target_score,
                    div_method=self.div_method,
                    div_samples=self.div_samples,
                )
                prior_logp = -0.5 * (x0 ** 2).sum(dim=[1, 2])
                prior_score = -x0
                N = x0.shape[1] * x0.shape[2]
                grad_x0 = (model_score - prior_score) / N
                ell = prior_logp + logdet
                ml_loss = -(ell + (x0 * grad_x0).sum(dim=[1, 2])).mean()

        if self.target_distribution is not None or self.weight_var_weight > 0:
            start = time.time()
            logprob0 = torch.zeros(base.shape[0], device=base.device)
            with torch.no_grad():
                xs_traj, logq_traj = self.flow_model.sample_logprob(
                    base, logprob0, div_method=self.div_method,
                    div_samples=self.div_samples, boxlength=None,
                    differentiable=False,
                )
                xs = xs_traj[-1]
                log_q = logq_traj[-1]
                if self.target_distribution is not None:
                    log_target = -self.target_distribution.potential(xs) / self.target_distribution.kT
                else:
                    log_target = torch.zeros_like(log_q)
                log_w = log_target - log_q
                log_w = log_w - log_w.max()
                w = torch.exp(log_w)
                w = w / w.sum()
                ess = 1.0 / (w.pow(2).sum())
                ess_norm = ess / w.shape[0]
                weight_var = w.var(unbiased=False)
            self.log('val_ess', ess_norm, on_step=False, on_epoch=True)
            if self.weight_var_weight > 0:
                self.log('val_weight_var', weight_var, on_step=False, on_epoch=True)

        v_squared_norm = (v**2).mean()
        total_loss = (
            self.cfm_weight * cfm_loss
            + self.ml_weight * ml_loss
            + self.weight_var_weight * weight_var
        )
        self.log("val_loss", total_loss / v_squared_norm, on_step=False, on_epoch=True)
        if self.ml_weight > 0:
            self.log('val_ml', ml_loss, on_step=False, on_epoch=True)
        self.log('val_cfm', cfm_loss, on_step=False, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

# same as above but we use the shortest Torus path to do flow matching.
class FlowTrainerTorus(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3, permute=False, sigma = 0, boxlength=None,
                 cfm_weight: float = 1.0, ml_weight: float = 0.0,
                 div_method: str = 'full_jacobian', div_samples: int = 1,
                 target_distribution=None, weight_var_weight: float = 0.0):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.permute = permute
        self.sigma = sigma  # Standard deviation for noise
        self.boxlength = boxlength if boxlength is not None else 100.0  # Default box length
        self.cfm_weight = cfm_weight
        self.ml_weight = ml_weight
        self.div_method = div_method
        self.div_samples = div_samples
        self.target_distribution = target_distribution
        self.weight_var_weight = weight_var_weight
        self.save_hyperparameters(ignore=["target_distribution"])

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
        if len(batch) == 3:
            base, target, weight = batch
        else:
            base, target = batch
            weight = torch.ones(base.shape[0], device=base.device)

        if self.permute:
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)
        vtorus = target - base
        to_subtract = ((torch.abs(vtorus) > 0.5 * self.boxlength)
                    * torch.sign(vtorus) * self.boxlength)
        vtorus = vtorus - to_subtract

        tr = t.view(-1, 1, 1)
        xt = base + tr * vtorus + self.sigma * torch.randn_like(base)
        xt = (xt + 0.5 * self.boxlength) % self.boxlength - 0.5 * self.boxlength

        vt = self.flow_model(xt, t)

        # Per-sample MSE
        per_sample_loss = F.mse_loss(vt, vtorus, reduction='none').mean(dim=[1, 2])

        # Apply optional weight
        weight = weight.to(per_sample_loss.device)
        cfm_loss = (weight * per_sample_loss).sum() / weight.sum()

        ml_loss = torch.tensor(0.0, device=base.device)
        weight_var = torch.tensor(0.0, device=base.device)
        if self.ml_weight > 0:
            x_traj, logp_traj = self.flow_model.sample_logprob(
                target,
                None,
                boxlength=self.boxlength,
                div_method=self.div_method,
                div_samples=self.div_samples,
                reverse=True,
                differentiable=True,
            )
            logdet = logp_traj[-1]
            log_target = torch.zeros_like(logdet)
            if self.target_distribution is not None:
                with torch.no_grad():
                    log_target = -self.target_distribution.potential(x_traj[-1]) / self.target_distribution.kT
            ml_loss = (log_target-PathGrad.apply(logdet, x_traj[-1])).mean()

        if self.target_distribution is not None or self.weight_var_weight > 0:
            start = time.time()
            logprob0 = torch.zeros(base.shape[0], device=base.device)
            if self.weight_var_weight > 0:
                xs_traj, logq_traj = self.flow_model.sample_logprob(
                    base, logprob0, boxlength=self.boxlength,
                    div_method=self.div_method, div_samples=self.div_samples,
                    differentiable=True,
                )
            else:
                with torch.no_grad():
                    xs_traj, logq_traj = self.flow_model.sample_logprob(
                        base, logprob0, boxlength=self.boxlength,
                        div_method=self.div_method, div_samples=self.div_samples,
                        differentiable=False,
                    )
            xs = xs_traj[-1]
            log_q = logq_traj[-1]
            with torch.no_grad():
                if self.target_distribution is not None:
                    log_target = -self.target_distribution.potential(xs) / self.target_distribution.kT
                else:
                    log_target = torch.zeros_like(log_q)
            log_w = log_target - log_q
            log_w = log_w - log_w.max()
            w = torch.exp(log_w)
            w = w / w.sum()
            ess = 1.0 / (w.pow(2).sum())
            ess_norm = ess / w.shape[0]
            weight_var = w.var(unbiased=False)
            self.log('train_ess', ess_norm, on_step=True, on_epoch=True)
            if self.weight_var_weight > 0:
                self.log('train_weight_var', weight_var, on_step=True, on_epoch=True)

        v_squared_norm = (vtorus ** 2).mean()
        total_loss = (
            self.cfm_weight * cfm_loss
            + self.ml_weight * ml_loss
            + self.weight_var_weight * weight_var
        )
        self.log("train_loss", total_loss / v_squared_norm, on_step=True, on_epoch=True)
        if self.ml_weight > 0:
            self.log('train_ml', ml_loss, on_step=True, on_epoch=True)
        self.log('train_cfm', cfm_loss, on_step=True, on_epoch=True)
        return total_loss

    def training_step_old(self, batch, batch_idx):
        base, target = batch

        if self.permute:
            # permute the batch to avoid symmetry issues
            #base = self.permute_batch(base)
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)  # shape: [batch]
        # we need to compute the shortest path in the torus
        vtorus = (target - base)
        to_subtract = ((torch.abs(vtorus)> 0.5 * self.boxlength)
                        * torch.sign(vtorus) * self.boxlength)
        vtorus = vtorus - to_subtract # right direction

        # Broadcast t to shape [batch, N, D] for interpolation
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base  + tr * vtorus + self.sigma * torch.randn_like(base)  # [batch, N, D] follow vtorus here and put back into the box.
        xt = xt % self.boxlength  # Ensure xt is within the box length
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(vtorus,vt)  # Example loss: minimize velocity magnitude
        v_squared_norm = (vtorus**2).mean()
        self.log("train_loss", loss/v_squared_norm, on_step = True, on_epoch = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Unpack weight if present
        if len(batch) == 3:
            base, target, weight = batch
        else:
            base, target = batch
            weight = torch.ones(base.shape[0], device=base.device)

        if self.permute:
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)

        # Handle torus correction
        vtorus = target - base
        to_subtract = ((torch.abs(vtorus) > 0.5 * self.boxlength)
                    * torch.sign(vtorus) * self.boxlength)
        vtorus = vtorus - to_subtract

        tr = t.view(-1, 1, 1)
        xt = base + tr * vtorus + self.sigma * torch.randn_like(base)
        xt = (xt + 0.5 * self.boxlength) % self.boxlength - 0.5 * self.boxlength

        vt = self.flow_model(xt, t)

        # Per-sample MSE loss
        per_sample_loss = F.mse_loss(vt, vtorus, reduction='none').mean(dim=[1, 2])
        weight = weight.to(per_sample_loss.device)
        cfm_loss = (weight * per_sample_loss).sum()/ weight.sum()

        ml_loss = torch.tensor(0.0, device=base.device)
        weight_var = torch.tensor(0.0, device=base.device)
        if self.ml_weight > 0:
            x_traj, logp_traj = self.flow_model.sample_logprob(
                target,
                None,
                boxlength=self.boxlength,
                div_method=self.div_method,
                div_samples=self.div_samples,
                reverse=True,
                differentiable=False,
            )
            ell = logp_traj[-1]
            base_logp = torch.zeros_like(ell)
            ml_loss = -(base_logp + ell).mean()

        if self.target_distribution is not None or self.weight_var_weight > 0:
            start = time.time()
            logprob0 = torch.zeros(base.shape[0], device=base.device)
            with torch.no_grad():
                xs_traj, logq_traj = self.flow_model.sample_logprob(
                    base, logprob0, boxlength=self.boxlength,
                    div_method=self.div_method, div_samples=self.div_samples,
                    differentiable=False,
                )
                xs = xs_traj[-1]
                log_q = logq_traj[-1]
                if self.target_distribution is not None:
                    log_target = -self.target_distribution.potential(xs) / self.target_distribution.kT
                else:
                    log_target = torch.zeros_like(log_q)
                log_w = log_target - log_q
                log_w = log_w - log_w.max()
                w = torch.exp(log_w)
                w = w / w.sum()
                ess = 1.0 / (w.pow(2).sum())
                ess_norm = ess / w.shape[0]
                weight_var = w.var(unbiased=False)
            self.log('val_ess', ess_norm, on_step=False, on_epoch=True)
            if self.weight_var_weight > 0:
                self.log('val_weight_var', weight_var, on_step=False, on_epoch=True)

        v_squared_norm = (vtorus ** 2).mean()
        total_loss = (
            self.cfm_weight * cfm_loss
            + self.ml_weight * ml_loss
            + self.weight_var_weight * weight_var
        )
        self.log("val_loss", total_loss / v_squared_norm, on_step=False, on_epoch=True)
        if self.ml_weight > 0:
            self.log('val_ml', ml_loss, on_step=False, on_epoch=True)
        self.log('val_cfm', cfm_loss, on_step=False, on_epoch=True)

        return total_loss


    def validation_step_old(self, batch, batch_idx):
        base, target = batch

        if self.permute:
            # permute the batch to avoid symmetry issues
            #base = self.permute_batch(base)
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)  # shape: [batch]

        # we need to compute the shortest path in the torus
        vtorus = (target - base)
        to_subtract = ((torch.abs(vtorus)> 0.5 * self.boxlength)
                        * torch.sign(vtorus) * self.boxlength)
        vtorus = vtorus - to_subtract # right direction

        # Broadcast t to shape [batch, N, D] for interpolation
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base  + tr * vtorus + self.sigma * torch.randn_like(base)  # [batch, N, D] follow vtorus here.
        # xt.requires_grad_()
        # v = target - base
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(vtorus,vt)  # Example loss: minimize velocity magnitude
        v_squared_norm = (vtorus**2).mean()
        self.log("val_loss", loss/v_squared_norm, on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

class DistillationTrainer(LightningModule):
    def __init__(self, flow_model, parent_model,  learning_rate=1e-3):
        super().__init__()
        self.flow_model = flow_model
        self.parent_model = parent_model
        self.parent_model.eval()  # Ensure parent model is in eval mode to avoid computing gradients
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.flow_model(x, t)

    def training_step(self, batch, batch_idx):
        base, target = batch
        t = torch.rand(base.shape[0], device=base.device)  # shape: [batch]
        # Broadcast t to shape [batch, N, D] for interpolation
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base * (1 - tr) + target * tr  # [batch, N, D]
        xt.requires_grad_()
        v = target - base
        vparent = self.parent_model(xt, t)  # Use parent model to get target velocity
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(v - vparent,vt)  # Example loss: minimize the discrepency between the flow model and parent model
        v_squared_norm = ((v - vparent) ** 2).mean()
        self.log("train_loss", loss/v_squared_norm, on_step = True, on_epoch = True)
        return loss

    # def on_after_backward(self):
    #     # Access gradient after backward
    #     if hasattr(self, "_last_xs") and self._last_xs.grad is not None:
    #         grad_mean = self._last_xs.grad.abs().mean().item()
    #         print(f"[Gradient check] ∂loss/∂xs.mean(): {grad_mean:.4e}")
    #     else:
    #         print("[Gradient check] No gradient on xs!")

    def validation_step(self, batch, batch_idx):
        base, target = batch
        t = torch.rand(base.shape[0], device=base.device)
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base * (1 - tr) + target * tr
        v = target - base
        vparent = self.parent_model(xt, t)  # Use parent model to get target velocity
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(v - vparent,vt)  # Example loss: minimize the discrepency between the flow model and parent model
        v_squared_norm = ((v - vparent) ** 2).mean()
        self.log("val_loss", loss/v_squared_norm, on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

# class DistillationTrainer(LightningModule):
#     def __init__(self, flow_model, learning_rate=1e-3):
#         super().__init__()
#         self.flow_model = flow_model
#         self.learning_rate = learning_rate
#         self.save_hyperparameters()

#     def forward(self, x, t):
#         return self.flow_model(x, t)

#     def training_step(self, batch, batch_idx):
#         x, t, y = batch
#         pred = self.flow_model(x, t)
#         loss = nn.MSELoss()(pred, y)
#         self.log("train_loss", loss.item(), on_step = False, on_epoch = True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, t, y = batch
#         pred = self.flow_model(x, t)
#         loss = nn.MSELoss()(pred, y)
#         self.log("val_loss", loss.item(), on_step = False, on_epoch = True)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

# we can implement the teacher student training here.
# Ie at chaque training stap just infer everything from the teacher
class TeacherTrainer(LightningModule):
    def __init__(self, flow_model, teacher, learning_rate=1e-3):
        super().__init__()
        self.flow_model = flow_model
        self.teacher = teacher
        self.teacher.eval()
        self.learning_rate = learning_rate
        self.teacher = teacher
        self.nbparticles = teacher.nbparticles
        self.dim = teacher.input_dim
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.flow_model(x, t)

    # need to modify that here.
    def training_step(self, batch, batch_idx):

        # gen random data
        nbsamples = self.nbsamples
        rd_data = torch.randn(nbsamples, self.nbparticles, self.dim) * 0.5
        rd_data = rd_data[torch.arange(rd_data.size(0)).unsqueeze(-1), torch.argsort(rd_data[:, :, 0], dim=1)] # sort by x
        x0 = rd_data.to(self.teacher.device)

        # propagate using teacher
        ts, traj = self.teacher.sample_traj(x0, n_steps=self.nbtimesteps)
        ts = repeat(ts, 't -> (t b) 1', b=traj.shape[1]).detach()
        traj = rearrange(traj, 't b p d -> (t b) p d').detach()
        field = self.teacher(traj, ts).detach()


        # compute the new flow field values and loss
        pred = self.flow_model(traj, ts)
        loss = nn.MSELoss()(pred, y)

        self.log("train_loss", loss.item(), on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        # exact same as training step
        nbsamples = self.nbsamples
        rd_data = torch.randn(nbsamples, self.nbparticles, self.dim) * 0.5
        rd_data = rd_data[torch.arange(rd_data.size(0)).unsqueeze(-1), torch.argsort(rd_data[:, :, 0], dim=1)] # sort by x
        x0 = rd_data.to(self.teacher.device)

        # propagate using teacher
        ts, traj = self.teacher.sample_traj(x0, n_steps=self.nbtimesteps)
        ts = repeat(ts, 't -> (t b) 1', b=traj.shape[1]).detach()
        traj = rearrange(traj, 't b p d -> (t b) p d').detach()
        field = self.teacher(traj, ts).detach()


        # compute the new flow field values and loss
        pred = self.flow_model(traj, ts)
        loss = nn.MSELoss()(pred, y)

        self.log("val_loss", loss.item(), on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)