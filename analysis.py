"""Load the saved result files and produce the tables and figures used in the notebooks."""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


def _plain_log_axis(ax):
    """Log y-axis with plain-number tick labels (1, 2, 5, 10, ...) instead of 2x10^1."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    lo, hi = ax.get_ylim()
    if hi / lo < 3:  # narrow range: label the minor ticks too
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OPTS = ["adam", "muon", "soap", "muon_spectral"]
NAME = {"adam": "Adam", "muon": "Muon", "soap": "SOAP", "muon_spectral": "Muon-spec"}
COLOR = {"adam": "#2a78d6", "muon": "#eb6834", "soap": "#1baf7a", "muon_spectral": "#eda100"}  # validated CVD-safe
STYLE = {"adam": "-", "muon": "--", "soap": "-.", "muon_spectral": ":"}                        # secondary encoding


def _load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def _nonempty(d):
    return {k: v for k, v in d.items() if v}


def main_runs(pde, results=RESULTS):
    return _nonempty({o: _load(f"{results}/{pde}/main_{o}_lr*.json") for o in OPTS})


def ft_runs(pde, results=RESULTS, base="adam_s0"):
    """Fine-tuning runs; `base` names the starting operator (e.g. "adam_s0", "l2adam_s0")."""
    return _nonempty({o: _load(f"{results}/{pde}/ft_{base}_{o}_lr*.json") for o in OPTS})


def sweep_table(pde, results=RESULTS):
    rows = [dict(optimizer=NAME[r["opt"]], lr=r["lr"], iterations=r["iters"], val_error_pct=100 * r["final_val"])
            for r in _load(f"{results}/{pde}/sweep_*.json")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["optimizer", "lr"])
    best = df.loc[df.groupby("optimizer")["val_error_pct"].idxmin()].set_index("optimizer")["lr"]
    df["chosen"] = [("<- best" if abs(l - best[o]) < 1e-12 else "") for o, l in zip(df.optimizer, df.lr)]
    return df.reset_index(drop=True).round({"val_error_pct": 3})


def results_table(pde, results=RESULTS):
    """Final test error (%) per test grid: mean ± std over seeds, plus lr and wall time."""
    out = []
    for o, rs in main_runs(pde, results).items():
        keys = list(rs[0]["final_test"])
        row = {"optimizer": NAME[o], "lr": rs[0]["lr"], "seeds": len(rs)}
        for k in keys:
            v = np.array([r["final_test"][k] for r in rs]) * 100
            row[f"test {k}"] = f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"
        row["time / run (min)"] = round(np.mean([r["train_time"] for r in rs]) / 60, 1)
        out.append(row)
    return pd.DataFrame(out).set_index("optimizer")


def ft_table(pde, results=RESULTS, base="adam_s0"):
    out = []
    for o, fs in ft_runs(pde, results, base).items():
        out.append({"optimizer": NAME[o], "lr": fs[0]["lr"], "instances": len(fs), "steps": fs[0]["iters"],
                    "start (%)": round(100 * np.mean([f["err0"] for f in fs]), 3),
                    "final (%)": round(100 * np.mean([f["final_err"] for f in fs]), 3),
                    "final per instance (%)": [round(100 * f["final_err"], 3) for f in fs],
                    "time / instance (s)": round(np.mean([f["time"] for f in fs]))})
    return pd.DataFrame(out).set_index("optimizer")


def plot_phase1(pde, results=RESULTS, x="it"):
    """Test error vs iteration (or minutes) on every test grid; mean over seeds."""
    runs = main_runs(pde, results)
    keys = list(next(iter(runs.values()))[0]["history"]["test"])
    fig, axs = plt.subplots(1, len(keys), figsize=(5 * len(keys), 3.8), constrained_layout=True)
    for ax, k in zip(np.atleast_1d(axs), keys):
        for o, rs in runs.items():
            L = min(len(r["history"]["it"]) for r in rs)
            xs = np.mean([r["history"]["it" if x == "it" else "time"][:L] for r in rs], 0)
            if x != "it":
                xs = xs / 60
            ys = np.mean([r["history"]["test"][k][:L] for r in rs], 0) * 100
            ax.plot(xs, ys, STYLE[o], color=COLOR[o], lw=2, marker="o", ms=3.5, label=NAME[o])
        _plain_log_axis(ax)
        ax.set_title(f"test grid {k}", loc="left")
        ax.set_xlabel("iteration" if x == "it" else "training time (min)")
        ax.set_ylabel("test rel. L² error (%)")
        ax.grid(True, which="both", color="#e5e5e5", lw=0.6)
        ax.legend(frameon=False)
    return fig


def plot_phase2(pde, results=RESULTS, x="it", base="adam_s0"):
    """Error vs reference during fine-tuning; geometric mean over instances (thin lines: each instance)."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for o, fs in ft_runs(pde, results, base).items():
        L = min(len(f["history"]["err"]) for f in fs)
        xs = np.mean([f["history"]["it" if x == "it" else "time"][:L] for f in fs], 0)
        E = np.array([f["history"]["err"][:L] for f in fs]) * 100
        for e in E:
            ax.plot(xs, e, STYLE[o], color=COLOR[o], lw=0.7, alpha=0.35)
        ax.plot(xs, np.exp(np.log(E).mean(0)), STYLE[o], color=COLOR[o], lw=2.2, label=NAME[o])
    _plain_log_axis(ax)
    ax.set_xlabel("fine-tuning step" if x == "it" else "fine-tuning time (s)")
    ax.set_ylabel("error vs reference (%)")
    ax.grid(True, which="both", color="#e5e5e5", lw=0.6)
    ax.legend(frameon=False)
    return fig


# ----------------------------------------------------------------------------------------------
# Experiment 1 (ansatz) and experiment 5 (hard solve from scratch)
# ----------------------------------------------------------------------------------------------
ANSATZ = {"fno": "FNO", "mlp_grid": "MLP + grid", "pinn": "PINN"}


def ansatz_table(results=os.path.join(RESULTS, "ansatz")):
    """Error (%) vs reference per grid and instance, next to the exact discrete minimizer ("floor")."""
    fl = json.load(open(f"{results}/floors.json"))
    rows = []
    for f in sorted(glob.glob(f"{results}/*_n*_s*.json")):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        m, rest = name.rsplit("_n", 1)
        n, s = rest.split("_s")
        rows.append(dict(grid=int(n), instance=int(s), method=ANSATZ[m], err=100 * r["final_err"],
                         floor=100 * fl[f"n{n}_s{s}"], minutes=r["time"] / 60))
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["grid", "instance"], columns="method", values="err")
    wide.insert(0, "floor", df.groupby(["grid", "instance"])["floor"].first())
    return wide[["floor", "FNO", "MLP + grid", "PINN"]].round(2)


def plot_ansatz(results=os.path.join(RESULTS, "ansatz")):
    t = ansatz_table(results).groupby("grid").mean()
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(t.index, t["floor"], color="#646b7a", lw=1.2, ls=(0, (4, 3)), marker="s", ms=5, label="floor")
    for col, (c, ls) in zip(["FNO", "MLP + grid", "PINN"],
                            [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-.")]):
        ax.plot(t.index, t[col], ls, color=c, lw=2, marker="o", ms=6, label=col)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(t.index))
    ax.set_xticklabels([f"{n}²" for n in t.index])
    ax.set_xlabel("PDE grid (space)")
    ax.set_ylabel("error vs reference (%)")
    ax.set_ylim(0, 50)
    ax.grid(True, color="#e5e5e5", lw=0.6)
    ax.legend(frameon=False)
    return fig


def hard_table(results=os.path.join(RESULTS, "hard_solve"), threshold=0.08, seeds=(1, 2)):
    """Optimizer comparison of notebook 08 (L2 residual) on the instances every optimizer ran."""
    fl = json.load(open(f"{results}/floors.json"))
    rows = []
    for o in OPTS:
        for f in sorted(glob.glob(f"{results}/{o}_s[0-9]*.json")):
            r = json.load(open(f))
            if r["seed"] not in seeds:
                continue
            h = r["history"]
            first = next((s for s, e in zip(h["step"], h["err"]) if e < threshold), None)
            rows.append({"optimizer": NAME[o], "instance": r["seed"], "lr": r["lr"],
                         "floor (%)": round(100 * fl[str(r["seed"])], 2),
                         "final (%)": round(100 * r["final_err"], 2),
                         f"first step < {100 * threshold:g}%": first,
                         "minutes": round(r["time"] / 60, 1)})
    df = pd.DataFrame(rows)
    df[f"first step < {100 * threshold:g}%"] = df[f"first step < {100 * threshold:g}%"].astype("Int64")
    return df.set_index(["instance", "optimizer"]).sort_index(level=0, sort_remaining=False)


def plot_hard(results=os.path.join(RESULTS, "hard_solve"), seeds=(1, 2)):
    fl = json.load(open(f"{results}/floors.json"))
    fig, axs = plt.subplots(1, len(seeds), figsize=(5.5 * len(seeds), 4), sharey=True, constrained_layout=True)
    for ax, s in zip(np.atleast_1d(axs), seeds):
        for o in OPTS:
            f = f"{results}/{o}_s{s}.json"
            if os.path.exists(f):
                h = json.load(open(f))["history"]
                ax.plot(h["step"], 100 * np.array(h["err"]), STYLE[o], color=COLOR[o], lw=2, label=NAME[o])
        ax.axhline(100 * fl[str(s)], color="#646b7a", lw=1.2, ls=(0, (4, 3)), label="floor")
        _plain_log_axis(ax)
        ax.set_title(f"instance {s}", loc="left")
        ax.set_xlabel("training step")
        ax.grid(True, which="both", color="#e5e5e5", lw=0.6)
    np.atleast_1d(axs)[0].set_ylabel("error vs reference (%)")
    np.atleast_1d(axs)[0].legend(frameon=False)
    return fig


# ----------------------------------------------------------------------------------------------
# Negative-Sobolev residual norms on Kolmogorov flow (notebook 09)
# ----------------------------------------------------------------------------------------------
NORMS = {"hs0": "L²", "hs1": "H⁻¹", "hs0.5": "H^(−1/2)"}
STARTS = {"adam_s0": "Adam-trained", "muon_s0": "Muon-trained", "soap_s0": "SOAP-trained"}


def kf_norms_table(results=os.path.join(RESULTS, "kf_norms")):
    """SOAP fine-tuning error (%) per starting operator, instance and residual norm."""
    fl = json.load(open(f"{results}/floors.json"))
    rows = []
    for d, norm in NORMS.items():
        for f in sorted(glob.glob(f"{results}/{d}/ft_*.json")):
            r = json.load(open(f))
            rows.append(dict(start=STARTS[r["base"]], instance=r["instance"], norm=norm,
                             err=100 * r["final_err"], floor=100 * fl[str(r["instance"])]))
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["start", "instance"], columns="norm", values="err")[list(NORMS.values())]
    wide.insert(0, "floor", df.groupby(["start", "instance"])["floor"].first())
    return wide.round(2)


def scratch_norms_table(results=os.path.join(RESULTS, "hard_solve")):
    """SOAP from scratch (notebook 08 setup) with L2 / H^-1 / H^-1/2 residuals, per instance."""
    fl = json.load(open(f"{results}/floors.json"))
    rows = []
    for s in sorted(int(k) for k in fl):
        row = {"instance": s, "floor": 100 * fl[str(s)]}
        for tag, norm in [("", "L²"), ("_hs1", "H⁻¹"), ("_hs0.5", "H^(−1/2)")]:
            f = f"{results}/soap{tag}_s{s}.json"
            row[norm] = 100 * json.load(open(f))["final_err"] if os.path.exists(f) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("instance").round(2)


def plot_scratch_norms(results=os.path.join(RESULTS, "hard_solve")):
    fl = json.load(open(f"{results}/floors.json"))
    seeds = sorted(int(k) for k in fl)
    fig, axs = plt.subplots(1, len(seeds), figsize=(3.6 * len(seeds), 3.6), sharey=True, constrained_layout=True)
    for ax, s in zip(np.atleast_1d(axs), seeds):
        for (tag, norm), (c, ls) in zip([("", "L²"), ("_hs1", "H⁻¹"), ("_hs0.5", "H^(−1/2)")],
                                        [("#2a78d6", "-"), ("#eb6834", "--"), ("#1baf7a", "-.")]):
            f = f"{results}/soap{tag}_s{s}.json"
            if os.path.exists(f):
                h = json.load(open(f))["history"]
                ax.plot(h["step"], 100 * np.array(h["err"]), ls, color=c, lw=2, label=norm)
        ax.axhline(100 * fl[str(s)], color="#646b7a", lw=1.2, ls=(0, (4, 3)), label="floor")
        _plain_log_axis(ax)
        ax.set_title(f"instance {s}", loc="left")
        ax.set_xlabel("training step")
        ax.grid(True, which="both", color="#e5e5e5", lw=0.6)
    np.atleast_1d(axs)[0].set_ylabel("error vs reference (%)")
    np.atleast_1d(axs)[0].legend(frameon=False)
    return fig
