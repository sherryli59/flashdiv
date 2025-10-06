from contextlib import contextmanager
import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt

@contextmanager
def requires_grad_(tensor: torch.Tensor, flag=True):
    old = tensor.requires_grad
    try:
        tensor.requires_grad_(flag)
        yield
    finally:
        tensor.requires_grad_(old)
def scan_batch(field, x_batch, t_scalar, max_points=8):
    """
    field(x,t) -> velocity
    Returns dictionary of diagnostics on a small subset of batch points.
    """
    device = x_batch.device
    B = min(x_batch.size(0), max_points)
    x = x_batch[:B].detach().clone().requires_grad_(True)
    t = torch.full((B,), float(t_scalar), device=device)

    sigma2, _ = power_iteration_spectral_norm(field, x, t)
    lam_max = sym_part_max_eig(field, x, t)
    fro = hutch_frobenius_norm(field, x, t, probes=8)
    lips_curve = finite_diff_lipschitz(field, x.detach(), t.detach())
    time_lips, time_curve = finite_diff_time_lipschitz(field, x.detach(), t.detach())

    return {
        "spectral_norm_est": sigma2,     # ~ ||J||_2
        "sym_part_lambda_max": lam_max,  # growth-rate bound
        "fro_norm_est": fro,             # >= ||J||_2
        "finite_diff_curve": lips_curve,  # list of (eps, L_eps)
        "time_lipschitz": time_lips,
        "time_lipschitz_curve": time_curve,
    }

def finite_diff_lipschitz(f, x, t, eps_list=(1e-1,5e-2,2e-2,1e-2,5e-3,2e-3,1e-3), trials=4):
    vals = []
    with torch.no_grad():
        for eps in eps_list:
            ratios = []
            for _ in range(trials):
                v = torch.randn_like(x)
                v = v / (v.norm() + 1e-12)
                fx  = f(x, t)
                fxp = f(x + eps * v, t)
                ratios.append((fxp - fx).norm().item() / (eps + 1e-12))
            vals.append((eps, float(np.median(ratios))))
    return vals  # list of (eps, approx local Lipschitz)


def finite_diff_time_lipschitz(f, x, t, eps_list=(5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3)):
    """Estimate sensitivity ||∂f/∂t|| via finite differences in time."""
    with torch.no_grad():
        x_eval = x.detach()
        if isinstance(t, torch.Tensor):
            base_t = t.detach()
        else:
            base_t = torch.full((x_eval.size(0),), float(t), device=x_eval.device)

        base_val = f(x_eval, base_t)
        curves = []
        for eps in eps_list:
            step = float(eps)
            t_offset = (base_t + step).clamp(0.0, 1.0)
            actual_step = (t_offset - base_t).abs().max().item()
            if actual_step < 1e-9:
                continue
            val_offset = f(x_eval, t_offset)
            diff_norm = (val_offset - base_val).norm().item()
            lips = diff_norm / (actual_step + 1e-12)
            curves.append((actual_step, lips))

        if not curves:
            return 0.0, []

        estimates = [val for _, val in curves]
        lips_est = float(np.median(estimates))
        curves = [(float(step), float(val)) for step, val in curves]

    return lips_est, curves


def hutch_frobenius_norm(f, x, t, probes=16):
    """
    E[||J||_F^2] ≈ E[||J z||^2] for z~N(0,I).
    Returns sqrt(mean ||Jz||^2) as an estimate of ||J||_F.
    """
    with requires_grad_(x, True):
        acc = 0.0
        for _ in range(probes):
            z = torch.randn_like(x)
            Jz = jvp(f, x, z, t)
            acc += (Jz**2).sum().item()
        mean_sq = acc / probes
        return (mean_sq**0.5)
    
def jvp(f, x, v, t):
    """Compute J(x) @ v via forward-over-reverse trick."""
    x_requires_grad = x.requires_grad
    x_eval = x if x_requires_grad else x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = f(x_eval, t)  # (B,P,D)
        flat_y = y.reshape(-1)
        g = grad(
            flat_y,
            x_eval,
            grad_outputs=torch.ones_like(flat_y),
            create_graph=True,
            retain_graph=True,
        )[0]
        s = (g * v).sum()
        Jv = grad(s, x_eval, retain_graph=True, create_graph=True)[0]
    return Jv if x_requires_grad else Jv.detach()

def vjp(f, x, w, t):
    """Compute J(x)^T @ w (plain reverse-mode)."""
    x_requires_grad = x.requires_grad
    x_eval = x if x_requires_grad else x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = f(x_eval, t)
        flat_y = y.reshape(-1)
        flat_w = w.reshape(-1).detach()
        JTw = grad(
            flat_y,
            x_eval,
            grad_outputs=flat_w,
            retain_graph=True,
            create_graph=True,
        )[0]
    return JTw if x_requires_grad else JTw.detach()

def power_iteration_spectral_norm(f, x, t, iters=30, tol=1e-5):
    """
    Estimate ||J(x)||_2 (local spectral norm) via power iteration
    using J^T J. Returns (sigma_est, converged).
    """
    v = torch.randn_like(x)
    v = v / (v.norm() + 1e-12)
    last = None
    for _ in range(iters):
        Jv = jvp(f, x, v, t).detach()
        JTJv = vjp(lambda z, tt: jvp(f, z, v, tt), x, Jv, t).detach()  # J^T (J v)
        JTJv_norm = JTJv.norm() + 1e-12
        v = JTJv / JTJv_norm
        sigma = Jv.norm() + 1e-12  # ≈ ||J v||
        if last is not None and (sigma - last).abs() < tol * max(1.0, last):
            return sigma.item(), True
        last = sigma
    return sigma.item(), False

def sym_part_max_eig(f, x, t, iters=30):
    """
    Estimate largest eigenvalue of the symmetric part S=(J+J^T)/2.
    This upper-bounds instantaneous growth rate (Lyapunov). If large ±,
    you’ll see stiffness/instability.
    Lanczos-like Rayleigh iteration using v -> Jv and JT@v.
    """
    with requires_grad_(x, True):
        v = torch.randn_like(x); v = v / (v.norm() + 1e-12)
        lam = None
        for _ in range(iters):
            Jv  = jvp(f, x, v, t)
            JT_v = vjp(f, x, v, t)
            Sv = 0.5 * (Jv + JT_v)                      # apply S to v
            v = Sv / (Sv.norm() + 1e-12)
            lam_new = (v * (0.5*(jvp(f, x, v, t) + vjp(f, x, v, t)))).sum() / (v.norm()**2 + 1e-12)
            if lam is not None and (lam_new - lam).abs() < 1e-5 * max(1.0, lam.abs()):
                break
            lam = lam_new.detach()
        return float(lam.item())


def _compute_state_norms(a: torch.Tensor, b: torch.Tensor, state_norm: str = "l2"):
    if state_norm == "l2":
        da = (a - b).reshape(a.shape[0], -1)
        nb = b.reshape(b.shape[0], -1)
        abs_ = torch.linalg.vector_norm(da, dim=1)
        rel_ = abs_ / (torch.linalg.vector_norm(nb, dim=1) + 1e-12)
        return abs_, rel_
    raise ValueError(f"Unknown state_norm '{state_norm}'")


@torch.no_grad()
def convergence_test_sample_logprob(
    flow,                  # object exposing sample_logprob(...)
    x,                      # (B, P, D)
    logprob,                # (B,)
    times=None,             # 1D tensor of times; if None your method fills [0,1]
    reverse=False,
    method: str = "rk4",
    h: float = 1/100,       # coarse step size
    factor: float = 2.0,    # compare h vs h/factor (usually 2.0)
    max_diff_abs_x: float = 1e-5,
    max_diff_rel_x: float = 1e-4,
    max_diff_abs_lp: float = 1e-6,
    max_diff_rel_lp: float = 1e-5,
    state_norm: str = "l2", # "l2" over flattened state per batch
    return_runs: bool = False,
    differentiable: bool = False,
    **kwargs,               # forwarded to sample_logprob (e.g., divergence impl flags)
):
    """
    Calls model.sample_logprob twice with (h) and (h/factor) and compares:
    - final state x_T (abs/rel per-batch L2)
    - final logprob lp_T (abs/rel per-batch)
    Returns a dict with pass/fail and max diffs.
    """

    # --- Run with coarse step size h
    xs_h, lp_h, *_ = flow.sample_logprob(
        x, logprob, times=times, reverse=reverse,
        differentiable=differentiable, return_traj=True,
        method=method, options={"step_size": h}, **kwargs
    )

    # --- Run with fine step size h/factor
    hf = h / factor
    xs_hf, lp_hf, *_ = flow.sample_logprob(
        x, logprob, times=times, reverse=reverse,
        differentiable=differentiable, return_traj=True,
        method=method, options={"step_size": hf}, **kwargs
    )

    # --- Compare finals (per-batch)
    x_abs, x_rel = _compute_state_norms(xs_h, xs_hf, state_norm)
    lp_abs = (lp_h - lp_hf).abs()                         # (B,)
    lp_rel = lp_abs / (lp_hf.abs() + 1e-12)               # (B,)

    # --- Aggregate & pass/fail
    x_abs_max  = x_abs.max().item()
    x_rel_max  = x_rel.max().item()
    lp_abs_max = lp_abs.max().item()
    lp_rel_max = lp_rel.max().item()

    passed = (
        (x_abs <= max_diff_abs_x).all()
        and (x_rel <= max_diff_rel_x).all()
        and (lp_abs <= max_diff_abs_lp).all()
        and (lp_rel <= max_diff_rel_lp).all()
    )

    report = {
        "passed": bool(passed),
        "state_diff_abs_max": x_abs_max,
        "state_diff_rel_max": x_rel_max,
        "logprob_diff_abs_max": lp_abs_max,
        "logprob_diff_rel_max": lp_rel_max,
        "h_coarse": float(h),
        "h_fine": float(hf),
        "method": method,
        "factor": float(factor),
    }

    if return_runs:
        report.update({
            "xs_coarse": xs_h, "lp_coarse": lp_h,
            "xs_fine": xs_hf, "lp_fine": lp_hf,
        })

    return report


@torch.no_grad()
def convergence_test_sample(
    flow,
    x,
    times=None,
    *,
    reverse=False,
    method: str = "rk4",
    h: float = 1/100,
    factor: float = 2.0,
    max_diff_abs_x: float = 1e-5,
    max_diff_rel_x: float = 1e-4,
    state_norm: str = "l2",
    return_runs: bool = False,
    differentiable: bool = False,
    **kwargs,
):
    """
    Runs flow.sample twice with step sizes (h) and (h/factor) and compares final states.
    """
    kwargs = dict(kwargs)
    kwargs.pop("differentiable", None)
    method = kwargs.pop("method", method)
    base_options = kwargs.pop("options", None)

    times = kwargs.pop("times", times)
    if times is None:
        times = torch.linspace(0.0, 1.0, 2, device=x.device, dtype=x.dtype)
    elif isinstance(times, torch.Tensor):
        times = times.to(device=x.device, dtype=x.dtype)
    else:
        times = torch.as_tensor(times, device=x.device, dtype=x.dtype)

    reverse = kwargs.pop("reverse", reverse)
    if reverse:
        times = torch.flip(times, dims=[0])

    hf = h / factor
    if base_options is not None:
        coarse_options = dict(base_options)
        fine_options = dict(base_options)
        coarse_options["step_size"] = h
        fine_options["step_size"] = hf
    else:
        coarse_options = {"step_size": h}
        fine_options = {"step_size": hf}

    xs_h, traj_h = flow.sample(
        x, times=times, return_traj=True,
        method=method, options=coarse_options, **kwargs
    )
    xs_hf, traj_hf = flow.sample(
        x, times=times, return_traj=True,
        method=method, options=fine_options, **kwargs
    )

    x_abs, x_rel = _compute_state_norms(xs_h, xs_hf, state_norm)
    x_abs_max = x_abs.max().item()
    x_rel_max = x_rel.max().item()

    passed = (
        (x_abs <= max_diff_abs_x).all()
        and (x_rel <= max_diff_rel_x).all()
    )

    report = {
        "passed": bool(passed),
        "state_diff_abs_max": x_abs_max,
        "state_diff_rel_max": x_rel_max,
        "logprob_diff_abs_max": None,
        "logprob_diff_rel_max": None,
        "h_coarse": float(h),
        "h_fine": float(hf),
        "method": method,
        "factor": float(factor),
    }

    if return_runs:
        report.update({
            "xs_coarse": xs_h,
            "xs_fine": xs_hf,
            "lp_coarse": None,
            "lp_fine": None,
            "traj_coarse": traj_h,
            "traj_fine": traj_hf,
        })

    return report
    
def find_stable_step(
    flow,
    x,
    lp=None,
    *,
    h0=1/50,
    min_h=1/1000,
    max_halves=6,
    rtol_x=1e-4,
    atol_x=1e-5,
    rtol_lp=1e-5,
    atol_lp=1e-6,
    prob: bool = True,
    **kw,
):
    h = h0
    for _ in range(max_halves+1):
        if prob:
            tester = getattr(flow, "convergence_test_sample_logprob", None)
            if tester is None:
                rep = convergence_test_sample_logprob(
                    flow, x, lp,
                    method="rk4", h=h, factor=2.0,
                    max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                    max_diff_abs_lp=atol_lp, max_diff_rel_lp=rtol_lp,
                    **kw,
                )
            else:
                rep = tester(
                    x, lp,
                    method="rk4", h=h, factor=2.0,
                    max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                    max_diff_abs_lp=atol_lp, max_diff_rel_lp=rtol_lp,
                    **kw,
                )
        else:
            rep = convergence_test_sample(
                flow, x,
                method="rk4", h=h, factor=2.0,
                max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                **kw,
            )
        if rep["passed"]:
            return h, rep
        h *= 0.5
        if h < min_h: break
    return h, rep  # last (failed) report so you can inspect

def convergence_sweep_sample_logprob(
    flow,
    x,
    lp=None,
    *,
    h0=1/50,
    num_halves=4,
    method="rk4",
    factor=2.0,
    rtol_x=1e-4,
    atol_x=1e-5,
    rtol_lp=1e-5,
    atol_lp=1e-6,
    prob: bool = True,
    **kw,
):
    """
    Runs your convergence test at h, h/2, h/4, ... and returns per-h reports.
    Set prob=False to compare samples without tracking logprob.
    """
    if prob and lp is None:
        raise ValueError("lp must be provided when prob=True.")

    hs, reports = [], []
    h = float(h0)
    for _ in range(max(1, num_halves)):
        print(f"Convergence test at h={h:.3e}")
        call_kw = dict(kw)
        if prob:
            rep = None
            tester = getattr(flow, "convergence_test_sample_logprob", None)
            if tester is not None:
                try:
                    rep = tester(
                        x, lp,
                        method=method, h=h, factor=factor,
                        max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                        max_diff_abs_lp=atol_lp, max_diff_rel_lp=rtol_lp,
                        **call_kw,
                    )
                except AttributeError:
                    rep = None  # fall back to module implementation
            if rep is None:
                rep = convergence_test_sample_logprob(
                    flow, x, lp,
                    method=method, h=h, factor=factor,
                    max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                    max_diff_abs_lp=atol_lp, max_diff_rel_lp=rtol_lp,
                    **call_kw,
                )
        else:
            rep = convergence_test_sample(
                flow, x,
                method=method, h=h, factor=factor,
                max_diff_abs_x=atol_x, max_diff_rel_x=rtol_x,
                **call_kw,
            )
        rep = dict(rep)  # ensure mutable copy
        rep["h"] = h
        reports.append(rep)
        hs.append(h)
        h *= 0.5
    return reports
    
def plot_convergence_sweep(
    reports,
    *,
    savepath="convergence_sweep.png",
    rtol_x=None,
    rtol_lp=None,
    title="RK4 step-doubling convergence",
):
    """
    Plots state/logprob relative max diffs vs. h in log-log.
    """
    hs = np.array([r["h"] for r in reports], dtype=float)
    rel_x = np.array([r.get("state_diff_rel_max", np.nan) for r in reports], dtype=float)
    rel_lp = np.array([r.get("logprob_diff_rel_max", np.nan) for r in reports], dtype=float)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(hs, rel_x, marker="o", label="state rel max")
    ax.loglog(hs, rel_lp, marker="s", label="logprob rel max")

    # Optional reference thresholds
    if rtol_x is not None:
        ax.axhline(rtol_x, ls="--", lw=1, color="tab:blue", alpha=0.5)
    if rtol_lp is not None:
        ax.axhline(rtol_lp, ls="--", lw=1, color="tab:orange", alpha=0.5)

    # Optional RK4 reference slope (~O(h^4) local error; step-doubling compares global error)
    # Draw a guideline with slope ~ h^4 relative to leftmost point for intuition:
    try:
        x0, y0 = hs[0], rel_x[0]
        guide = y0 * (hs / x0)**4
        ax.loglog(hs, guide, ls=":", color="gray", alpha=0.5, label="h^4 guide")
    except Exception:
        pass

    ax.set_xlabel("step size h")
    ax.set_ylabel("max relative diff (h vs. h/2)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)
    return savepath
