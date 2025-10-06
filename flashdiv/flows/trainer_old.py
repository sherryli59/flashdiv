
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from einops import repeat, rearrange, reduce

class FlowTrainer(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3, permute=False, sigma = 0):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.permute = permute
        self.sigma = sigma  # Standard deviation for noise
        self.save_hyperparameters()

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
        base, target = batch

        if self.permute:
            # permute the batch to avoid symmetry issues
            #base = self.permute_batch(base)
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)  # shape: [batch]
        # Broadcast t to shape [batch, N, D] for interpolation
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base * (1 - tr) + target * tr + self.sigma * torch.randn_like(base)  # [batch, N, D]
        # xt.requires_grad_()
        v = target - base
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(v,vt)  # Example loss: minimize velocity magnitude
        v_squared_norm = (v**2).mean()
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

        if self.permute:
            # permute the batch to avoid symmetry issues
            #base = self.permute_batch(base)
            target = self.permute_batch(target)

        t = torch.rand(base.shape[0], device=base.device)
        tr = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        xt = base * (1 - tr) + target * tr + self.sigma * torch.randn_like(base)  # [batch, N, D]
        v = target - base
        vt = self.flow_model(xt, t)
        loss = nn.MSELoss()(v,vt)  # Example loss: minimize velocity magnitude
        v_squared_norm = (v**2).mean()
        self.log("val_loss", loss/v_squared_norm, on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

# same as above but we use the shortest Torus path to do flow matching.
class FlowTrainerTorus(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3, permute=False, sigma = 0, boxlength=None):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.permute = permute
        self.sigma = sigma  # Standard deviation for noise
        self.boxlength = boxlength if boxlength is not None else 100.0  # Default box length
        self.save_hyperparameters()

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
        #weight = weight / torch.clamp(weight.sum(), min=1e-6) 
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
        loss = (weight * ((vtorus - vt)**2).flatten(1).mean(1)).sum()
        v_squared_norm_w = (weight * (vtorus**2).flatten(1).mean(1)).sum()
        self.log("train_loss", loss/v_squared_norm_w, on_step = True, on_epoch = True)
        return loss
    

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
        weight = weight / torch.clamp(weight.sum(), min=1e-6) 
        if self.permute:
            # permute the batch to avoid symmetry issues
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
        loss = (weight * ((vtorus - vt)**2).flatten(1).mean(1)).sum()
        v_squared_norm_w = (weight * (vtorus**2).flatten(1).mean(1)).sum()
        self.log("val_loss", loss/v_squared_norm_w, on_step = False, on_epoch = True)
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