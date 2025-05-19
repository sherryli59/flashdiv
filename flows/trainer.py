
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from einops import repeat, rearrange, reduce

class FlowTrainer(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3):
        super().__init__()
        self.factorizedflow = flow_model
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.factorizedflow(x, t)

    def training_step(self, batch, batch_idx):
        base, target = batch
        t = torch.rand((base.shape[0],1), device=base.device)
        tr = repeat(t, 'b 1 -> b l k', l=base.shape[1], k=base.shape[2])
        xt = base * (1 - tr) + target * tr
        v = target - base
        vt = self.factorizedflow(xt, t)
        loss = nn.MSELoss()(v,vt)  # Example loss: minimize velocity magnitude
        self.log("train_loss", loss, on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        base, target = batch
        t = torch.rand((base.shape[0],1), device=base.device)
        tr = repeat(t, 'b 1 -> b l k', l=base.shape[1], k=base.shape[2])
        xt = base * (1 - tr) + target * tr
        v = target - base
        vt = self.factorizedflow(xt, t)
        loss = nn.MSELoss()(v,vt)  # Example loss: minimize velocity magnitude
        self.log("val_loss", loss, on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.factorizedflow.parameters(), lr=self.learning_rate)

class DistillationTrainer(LightningModule):
    def __init__(self, flow_model, learning_rate=1e-3):
        super().__init__()
        self.flow_model = flow_model
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x, t):
        return self.flow_model(x, t)

    def training_step(self, batch, batch_idx):
        x, t, y = batch
        pred = self.flow_model(x, t)
        loss = nn.MSELoss()(pred, y)
        self.log("train_loss", loss.item(), on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, y = batch
        pred = self.flow_model(x, t)
        loss = nn.MSELoss()(pred, y)
        self.log("val_loss", loss.item(), on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.flow_model.parameters(), lr=self.learning_rate)

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