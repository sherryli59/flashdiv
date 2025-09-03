import sys
import pathlib
import torch

# add project path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'lj_reflow_template'))
from flashdiv.flows.flow_net_torchdiffeq import FlowNet


class ZeroFlow(FlowNet):
    def forward(self, x, t):
        return x * 0

    def divergence(self, x, t, **kwargs):
        return torch.zeros(x.shape[0], device=x.device)


def test_sample_logprob_zero_flow():
    model = ZeroFlow()
    x0 = torch.randn(4, 2, 1)
    logp0 = torch.zeros(x0.shape[0])
    xs, logp = model.sample_logprob(x0, logp0)
    assert torch.allclose(xs[-1], x0)
    assert torch.allclose(logp[-1], logp0)


def test_integrate_augmented_adj_zero_flow():
    model = ZeroFlow()
    x1 = torch.randn(3, 2, 1)
    adj1 = torch.randn_like(x1)
    x0, score, logdet = model.integrate_augmented_adj(x1, adj1)
    assert torch.allclose(x0, x1)
    assert torch.allclose(score, adj1)
    assert torch.allclose(logdet, torch.zeros(x1.shape[0]))
