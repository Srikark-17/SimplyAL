#!/usr/bin/env python
"""Fill the SimplyAL manuscript with the results of simulate_al.py.

Reads the folders written by simulate_al.py and writes into --tex-dir:
  results.tex                          values that SimplyAL_revised.tex reads
  figures/active_learning_curves.pdf   Figure 5 (least confidence vs random)
  results_report.txt                   warnings, generated sentences, every value
  letter_snippets.txt                  text for the [[NAMES]] in the response letter
With --repo it also writes the learning-rate presets into the SimplyAL frontend.

run_all.sh calls it at the end; it can be rerun on its own at any time:
  python backend/fill_results.py --runs runs/main --cpu-runs runs/cpu \\
      --csv /data/fundus.csv --label-column annotation --path-column file_path \\
      --tex-dir /path/to/paper --repo .
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = {"lr": 1e-3, "episode_size": 16, "batch_size": 16, "epochs": 5}
MODELS = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "dinov2": "DINOv2"}
ROWS = {  # Table 4: key -> (strategy, settings changed from DEFAULT, description)
    "default": ("least_confidence", {}, "the default settings"),
    "random": ("random", {}, "random sampling"),
    "margin": ("margin", {}, "margin sampling"),
    "entropy": ("entropy", {}, "entropy sampling"),
    "coreset": ("diversity", {}, "core-set sampling"),
    "lr4": ("least_confidence", {"lr": 1e-4}, "a learning rate of $10^{-4}$"),
    "lr5": ("least_confidence", {"lr": 1e-5}, "a learning rate of $10^{-5}$"),
    "ep8": ("least_confidence", {"episode_size": 8, "batch_size": 8}, "8 images per episode"),
    "ep32": ("least_confidence", {"episode_size": 32, "batch_size": 32}, "32 images per episode"),
    "epochs3": ("least_confidence", {"epochs": 3}, "3 epochs per episode"),
    "epochs10": ("least_confidence", {"epochs": 10}, "10 epochs per episode"),
}
RESNET50_ONLY = {"ep8", "ep32", "epochs3", "epochs10"}
STRATEGY_ROWS = ["default", "random", "margin", "entropy", "coreset"]
LR_ROWS = {"default": 1e-3, "lr4": 1e-4, "lr5": 1e-5}
SOURCES = [  # key, fragments of the path or source name; checked in order (mBRSET before BRSET)
    ("mbrset", ["mbrset"]), ("brset", ["brset"]), ("aizawl", ["aizawl"]),
    ("drishti", ["drishti"]), ("kcg", ["kcg"]), ("lesav", ["les-av", "les_av", "lesav"]),
    ("rfmid", ["rfmid", "riadd"]), ("riga", ["riga"]),
]
LATENCY = [("n", "n_labeled"), ("train", "train_s"), ("val", "validation_s"),
           ("preview", "preview_s"), ("wait1", "wait1_s"), ("wait2", "wait2_s")]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
SNIPPETS = ["N-TEST", "N-SEEDS", "PLATEAU-SENTENCE", "GUIDANCE", "SENSITIVITY",
            "GPU", "WAIT1-RANGE", "WAIT2-RANGE"]


class Filled:
    def __init__(self):
        self.values, self.snippets, self.warnings, self.notes = {}, {}, [], []

    def set(self, key, value):
        self.values[key] = value

    def warn(self, message):
        self.warnings.append(message)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, required=True, help="simulate_al.py output folder (GPU runs)")
    p.add_argument("--cpu-runs", type=Path, help="output folder of the CPU run (optional)")
    p.add_argument("--csv", type=Path, help="labeled CSV used for the runs (class and source counts)")
    p.add_argument("--label-column", default="annotation")
    p.add_argument("--path-column", default="file_path")
    p.add_argument("--source-column", help="CSV column naming each image's source dataset (optional)")
    p.add_argument("--tex-dir", type=Path, required=True, help="paper folder with SimplyAL_revised.tex")
    p.add_argument("--repo", type=Path, help="SimplyAL repository: write the learning-rate presets")
    return p.parse_args()


def join_and(items):
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def tex_escape(text):
    for ch in "&%$#_{}":
        text = text.replace(ch, "\\" + ch)
    return text


def tex_lr(value):
    return f"$10^{{{int(round(np.log10(value)))}}}$"


def plain(tex):
    """LaTeX sentence -> plain text for the response letter."""
    s = re.sub(r"Section~\\ref\{sec:sensitivity\}", "Section 3.4", tex)
    s = s.replace("\\cite{munjal2022robust}", "(Munjal et al., 2022)")
    s = re.sub(r"\$10\^\{(-?\d+)\}\$", r"10^\1", s)
    s = re.sub(r"\$([\d.]+) \\pm ([\d.]+)\$", "\\1 ± \\2", s)
    return s.replace("\\%", "%").replace("~", " ").strip()


def mean_sd(mean, sd):
    if pd.isna(mean):
        return None
    if pd.isna(sd):
        return f"{100 * mean:.1f}"
    return f"${100 * mean:.1f} \\pm {100 * sd:.1f}$"


def seconds(value):
    if pd.isna(value):
        return "--"
    return f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def count(value):
    return "--" if pd.isna(value) else f"{int(round(value))}"


def matches(df, model, strategy, changes=None):
    """Rows of one configuration: DEFAULT settings, with `changes` applied."""
    s = dict(DEFAULT, **(changes or {}))
    keep = (df.strategy == strategy) & np.isclose(df.lr.astype(float), s["lr"])
    for k in ("episode_size", "batch_size", "epochs"):
        keep &= df[k].astype(int) == s[k]
    if model is not None:
        keep &= df.model == model
    return df[keep]


def read_runs(folder):
    files = {name: folder / name for name in ("results.csv", "summary.csv", "plateau.csv")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        sys.exit("missing " + ", ".join(missing) + ": run simulate_al.py (run_all.sh) first")
    env_path = folder / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    return (pd.read_csv(files["results.csv"]), pd.read_csv(files["summary.csv"]),
            pd.read_csv(files["plateau.csv"]), env)


def fill_env(F, env):
    gpu = env.get("gpu")
    parts = []
    if gpu:
        mem = env.get("gpu_memory_gb")
        parts.append(gpu + (f" ({mem:.0f} GB)" if mem else ""))
        F.snippets["GPU"] = gpu
    else:
        F.warn("env.json shows no GPU: the main runs did not use CUDA")
    cpu = env.get("cpu_model") or env.get("processor")
    if cpu:
        parts.append(cpu + (f", {env['cpu_count']} logical cores" if env.get("cpu_count") else ""))
    if env.get("ram_gb"):
        parts.append(f"{env['ram_gb']:.0f} GB RAM")
    if env.get("os") or env.get("platform"):
        parts.append(env.get("os") or env.get("platform"))
    if parts:
        F.set("env-hardware", tex_escape("; ".join(parts)))
    for key in ("python", "torch", "torchvision", "timm"):
        if env.get(key):
            F.set(f"env-{key}", tex_escape(str(env[key])))
    F.set("env-cuda", tex_escape(str(env["cuda"])) if env.get("cuda") else "not used")
    F.set("env-gpu", tex_escape(gpu) if gpu else "a CPU")


def fill_sizes(F, res, env):
    lc = matches(res, None, "least_confidence")
    if lc.empty:
        F.warn("no default least-confidence runs in results.csv")
        return
    first = lc[lc.episode == 0]
    n_seed = int(first.n_labeled.mode().iloc[0])
    nontest = int(round((first.n_labeled / first.pct_labeled * 100).median()))
    F.set("n-seed", f"{n_seed:,}")
    F.set("n-nontest", f"{nontest:,}")
    n_images = env.get("n_images")
    if n_images:
        F.set("n-test", f"{int(n_images) - nontest:,}")
        F.snippets["N-TEST"] = f"{int(n_images) - nontest:,}"
        if int(n_images) != 1020:
            F.warn(f"{n_images} images were loaded, but the paper says 1,020 (abstract, "
                   "Section 3.2, Table 3): check the CSV or update those numbers")
    seeds = lc.groupby("model").seed.nunique()
    n = int(seeds.min())
    F.set("n-seeds", str(n))
    F.snippets["N-SEEDS"] = WORDS.get(n, str(n))
    if n < 3:
        F.warn(f"only {n} seed(s) for some backbones; the paper reports {n}")
    absent = [MODELS[m] for m in MODELS if m not in seeds.index]
    if absent:
        F.warn("no default least-confidence runs for " + ", ".join(absent))


def curve(rows):
    return rows.groupby("episode").agg(x=("pct_labeled", "mean"), acc=("test_acc", "mean")).sort_index()


def plateau_point(F, plat, strategy):
    p = matches(plat, None, strategy)
    if p.empty:
        return None
    models = set(str(p.models.iloc[0]).split(","))
    if models != set(MODELS):
        F.warn(f"the {strategy} plateau averages only {', '.join(sorted(models))}")
    value = p.plateau_pct_labeled.iloc[0]
    return None if pd.isna(value) else float(value)


def fill_efficiency(F, res, plat):
    lc_rows = matches(res, None, "least_confidence")
    rnd_rows = matches(res, None, "random")
    if lc_rows.empty:
        return None
    lc = curve(lc_rows)
    p_lc = plateau_point(F, plat, "least_confidence")
    p_rnd = plateau_point(F, plat, "random") if not rnd_rows.empty else None
    sentences = []
    if p_lc is None:
        F.warn("least-confidence sampling did not meet the plateau criterion within the budget: "
               "the abstract and Sections 3.3, 4 and 5 show red text to rewrite by hand")
        sentences.append("\\fillin{No plateau within the budget: describe the curves here.}")
        F.set("fig5-plateau-note", "")
    else:
        episode = int(lc.index[np.argmin(np.abs(lc.x.to_numpy() - p_lc))])
        F.set("plateau-lc", f"{p_lc:.0f}")
        first = (f"With least-confidence sampling, performance plateaued after {p_lc:.0f}\\% "
                 f"of the images were labeled (about {episode} episodes)")
        if p_rnd is not None:
            first += f", and with random sampling after {p_rnd:.0f}\\%"
        elif not rnd_rows.empty:
            first += f", whereas random sampling had not plateaued by {curve(rnd_rows).x.iloc[-1]:.0f}\\%"
        sentences.append(first + ".")
        at = lc_rows[lc_rows.episode == episode].groupby("model").test_acc.mean()
        accs = [f"{100 * at[m]:.1f}\\% for {name}" for m, name in MODELS.items() if m in at.index]
        if accs:
            sentences.append("At that point, mean test accuracy was " + join_and(accs) + ".")
        F.set("fig5-plateau-note", "The vertical line marks the plateau point of least-confidence sampling.")
        F.snippets["PLATEAU-SENTENCE"] = (
            f"With this evaluation, performance plateaus after {p_lc:.0f}% of the images are "
            "labeled, compared with 35% in the original submission.")
    case, savings = None, None
    if rnd_rows.empty:
        F.warn("no random-sampling runs: Section 3.3 has no comparison with random sampling")
    else:
        rnd = curve(rnd_rows)
        both = lc.join(rnd, lsuffix="_lc", rsuffix="_rnd", how="inner")
        diff = 100 * (both.acc_lc - both.acc_rnd)
        target = rnd.acc.iloc[-1]
        reached = lc[lc.acc >= target]
        if len(reached):
            savings = 100 * (1 - reached.x.iloc[0] / rnd.x.iloc[-1])
        if savings is not None and savings >= 5:
            case = "lc"
            sentences.append(f"Least-confidence sampling reached the mean final accuracy of random sampling "
                             f"({100 * target:.1f}\\%) with {savings:.0f}\\% fewer labels.")
        elif diff.mean() < -1:
            case = "random"
            sentences.append(f"Random sampling matched or exceeded least-confidence sampling in this "
                             f"experiment, by {-diff.mean():.1f} percentage points on average, consistent "
                             "with reports that gains over random sampling can be small with pretrained "
                             "backbones \\cite{munjal2022robust}.")
        else:
            case = "similar"
            sentences.append(f"Least-confidence and random sampling reached similar accuracy, differing "
                             f"by at most {diff.abs().max():.1f} percentage points at any annotation level, "
                             "consistent with reports that gains over random sampling can be small with "
                             "pretrained backbones \\cite{munjal2022robust}.")
    F.set("efficiency-findings", " ".join(sentences))
    clause = (f"least-confidence sampling needed {savings:.0f}\\% fewer labels than random sampling "
              "to reach the same accuracy") if case == "lc" else None
    F.set("abstract-random", f"Least-confidence sampling reached the final accuracy of random sampling "
                             f"with {savings:.0f}\\% fewer labels. " if case == "lc" else "")
    F.set("conclusion-random", f", and {clause}" if clause else "")
    if p_lc is not None:
        F.set("impact-findings", f"performance plateaued after approximately {p_lc:.0f}\\% of the "
                                 "images were labeled" + (f", and {clause}" if clause else ""))
    return p_lc


def fill_sensitivity(F, summ):
    at50, absent = {}, []
    for row, (strategy, changes, _) in ROWS.items():
        for m in MODELS:
            if row in RESNET50_ONLY and m != "resnet50":
                continue
            hit = matches(summ, m, strategy, changes)
            if hit.empty:
                absent.append(f"{row} ({MODELS[m]})")
                continue
            r = hit.iloc[0]
            for budget in (25, 50):
                value = mean_sd(r.get(f"acc_at_{budget}_mean"), r.get(f"acc_at_{budget}_sd"))
                if value:
                    F.set(f"sens-{row}-{m}-{budget}", value)
            if not pd.isna(r.get("acc_at_50_mean")):
                sd = r.get("acc_at_50_sd")
                at50[(row, m)] = (100 * r["acc_at_50_mean"], 0.0 if pd.isna(sd) else 100 * sd)
    if absent:
        F.warn("Table 4 has no runs for " + ", ".join(absent))

    def get(row, m):
        return at50[(row, m)][0] if (row, m) in at50 else None

    s = []
    spans = []
    for m in MODELS:
        v = [get(r, m) for r in STRATEGY_ROWS]
        if None not in v:
            spans.append(max(v) - min(v))
    if spans:
        s.append(f"At 50\\% annotation, mean test accuracy varied by at most {max(spans):.1f} "
                 "percentage points across the five acquisition strategies.")
    lr_span = {}
    for m in MODELS:
        v = [get(r, m) for r in LR_ROWS]
        if None not in v:
            lr_span[m] = (min(v), max(v))
    if lr_span:
        m = max(lr_span, key=lambda k: lr_span[k][1] - lr_span[k][0])
        s.append(f"The learning rate mattered most for {MODELS[m]}, whose accuracy ranged from "
                 f"{lr_span[m][0]:.1f}\\% to {lr_span[m][1]:.1f}\\%.")
    v = [get(r, "resnet50") for r in ("default", "ep8", "ep32", "epochs3", "epochs10")]
    if None not in v:
        s.append(f"For ResNet-50, changing the images or epochs per episode moved accuracy by at most "
                 f"{max(v) - min(v):.1f} points.")
    gaps = {}
    for m in MODELS:
        d = get("default", m)
        if d is None:
            continue
        best = max((r for r in ROWS if get(r, m) is not None), key=lambda r: get(r, m))
        gaps[m] = (get(best, m) - d, best)
    if gaps:
        near = {m: g for m, g in gaps.items() if g[0] <= 1.0}
        far = {m: g for m, g in gaps.items() if g[0] > 1.0}
        if not far:
            s.append(f"With the default settings, every backbone was within "
                     f"{max(g[0] for g in gaps.values()):.1f} points of its best setting.")
        else:
            gains = join_and([f"{MODELS[m]} gained {g:.1f} points with {ROWS[row][2]}"
                              for m, (g, row) in far.items()])
            if near:
                verb, pronoun = ("was", "its") if len(near) == 1 else ("were", "their")
                s.append(f"With the default settings, {join_and(MODELS[m] for m in near)} {verb} within "
                         f"{max(g[0] for g in near.values()):.1f} points of {pronoun} best setting, "
                         f"whereas {gains}.")
            else:
                s.append(f"Every backbone improved on the defaults: {gains}.")
    if s:
        F.set("sensitivity-findings", " ".join(s))
        F.snippets["SENSITIVITY"] = plain(" ".join(s))
    return at50


def fill_presets(F, at50):
    """Adopt another learning rate for a backbone only if it beats the default by more
    than one standard deviation and more than 1 percentage point at 50% annotation."""
    presets, considered = {}, []
    for m in MODELS:
        cand = {r: at50[(r, m)] for r in LR_ROWS if (r, m) in at50}
        if "default" not in cand or len(cand) < 2:
            continue
        considered.append(m)
        best = max(cand, key=lambda r: cand[r][0])
        gain = cand[best][0] - cand["default"][0]
        if best != "default" and gain > max(cand[best][1], cand["default"][1], 1.0):
            presets[m] = LR_ROWS[best]
    F.set("default-lr", f"{presets.get('resnet50', 1e-3):g}")
    if not considered:
        F.warn("no learning-rate runs: Section 2.4 shows red text")
        return presets
    if presets:
        listing = join_and([f"{tex_lr(v)} for {MODELS[m]}" for m, v in presets.items()])
        rest = [MODELS[m] for m in considered if m not in presets]
        tail = f"; {join_and(rest)} {'keeps' if len(rest) == 1 else 'keep'} 0.001" if rest else ""
        text = ("Selecting a backbone loads the learning rate that performed best for it in the "
                f"sensitivity analysis (Section~\\ref{{sec:sensitivity}}): {listing}{tail}. ")
    else:
        text = ("In the sensitivity analysis (Section~\\ref{sec:sensitivity}), no other learning rate "
                "improved on the default by more than one standard deviation or 1 percentage point "
                "for any backbone. ")
    F.set("guidance-lr", text)
    F.snippets["GUIDANCE"] = plain(text)
    return presets


def fill_latency(F, summ, cpu_summ, cpu_env):
    waits = {"wait1": [], "wait2": []}
    for m in MODELS:
        hit = matches(summ, m, "least_confidence")
        if hit.empty:
            continue
        r = hit.iloc[0]
        for when in ("first", "last"):
            for key, column in LATENCY:
                value = r.get(f"{when}_{column}_mean")
                F.set(f"lat-{m}-{when}-{key}", count(value) if key == "n" else seconds(value))
                if key in waits and not pd.isna(value):
                    waits[key].append(value)

    def span(xs):
        f = (lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}")
        return f"{f(min(xs))}–{f(max(xs))}"

    for key, name in (("wait1", "WAIT1-RANGE"), ("wait2", "WAIT2-RANGE")):
        if waits[key]:
            F.snippets[name] = span(waits[key])
    hit = matches(cpu_summ, "resnet18", "least_confidence") if cpu_summ is not None else pd.DataFrame()
    if not hit.empty:
        r = hit.iloc[0]
        for key, column in LATENCY:
            value = r.get(f"first_{column}_mean")
            F.set(f"lat-cpu-{key}", count(value) if key == "n" else seconds(value))
        cpu = cpu_env.get("cpu_model") or cpu_env.get("processor") or "the server's CPU"
        F.set("lat-cpu-note", f"The CPU row is a single run on {tex_escape(cpu)}.")
    else:
        for key, _ in LATENCY:
            F.set(f"lat-cpu-{key}", "--")
        F.set("lat-cpu-note", "The CPU row was not measured.")
        F.warn("no CPU run: the CPU row of Table 5 shows dashes; run stage 7 of run_all.sh or delete the row")


def fill_data(F, args):
    if not args.csv or not args.csv.exists():
        F.warn("no --csv: class and source counts in Table 3 stay red")
        return
    df = pd.read_csv(args.csv)
    if args.label_column not in df.columns:
        F.warn(f"{args.csv} has no column {args.label_column!r}: class counts stay red")
        return
    df = df.dropna(subset=[args.label_column])
    counts = df[args.label_column].astype(str).str.strip().value_counts().sort_index()
    F.set("data-classes", join_and([f"{n:,} {tex_escape(name)}" for name, n in counts.items()]))
    if len(counts) == 2 and counts.max() > 1.02 * counts.min():
        F.warn(f"classes are not evenly split ({dict(counts)}): edit 'Labels were split evenly' in Section 3.2")
    column = args.source_column or args.path_column
    if column not in df.columns:
        F.warn(f"{args.csv} has no column {column!r}: source counts stay red")
        return
    text = df[column].astype(str).str.lower()
    source = pd.Series(None, index=df.index, dtype=object)
    for key, fragments in SOURCES:
        hit = source.isna() & text.apply(lambda s: any(f in s for f in fragments))
        source[hit] = key
    for key, _ in SOURCES:
        n = int((source == key).sum())
        if n:
            F.set(f"src-{key}", f"{n:,}")
    if source.isna().any():
        F.warn(f"{int(source.isna().sum())} images matched no source dataset in column {column!r}. "
               "Add a source column to the CSV and pass --source-column, or put lines such as "
               "\\setres{src-brset}{300} in results_manual.tex")


def plot_figure5(res, plateau_x, path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    colors = {"dinov2": "C0", "resnet50": "C1", "resnet18": "C2"}
    styles = {"least_confidence": ("-", "least confidence"), "random": ("--", "random")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in zip(axes, ("test_acc", "test_f1"), ("Test accuracy (%)", "Weighted F1-score (%)")):
        for m in ("dinov2", "resnet50", "resnet18"):
            for strategy, (ls, name) in styles.items():
                rows = matches(res, m, strategy)
                if rows.empty:
                    continue
                c = rows.groupby("episode").agg(x=("pct_labeled", "mean"), mu=(metric, "mean"),
                                                sd=(metric, "std")).sort_index()
                y, sd = 100 * c.mu, 100 * c.sd.fillna(0)
                ax.plot(c.x, y, ls, color=colors[m], lw=1.5, label=f"{MODELS[m]}, {name}")
                ax.fill_between(c.x, y - sd, y + sd, color=colors[m], alpha=0.12, lw=0)
        if plateau_x is not None:
            ax.axvline(plateau_x, color="red", ls=":", lw=1.2, label="Plateau, least confidence")
        ax.set_xlabel("Non-test images labeled (%)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def write_presets(F, repo, presets):
    path = repo / "frontend" / "src" / "app" / "annotation.jsx"
    if not path.exists():
        F.warn(f"{path} not found: set LR_PRESETS by hand")
        return
    src = path.read_text()
    line = "const LR_PRESETS = { " + ", ".join(f"{m}: {presets.get(m, 1e-3):g}" for m in MODELS) + " };"
    new, n = re.subn(r"const LR_PRESETS = \{[^}]*\};", line, src)
    if n != 1:
        F.warn("LR_PRESETS not found in annotation.jsx: is the v1.1 patch applied?")
        return
    if "resnet50" in presets:
        new, k = re.subn(r"(useState\(\{\s*strategy: \"plateau\",[^\n]*\n\s*initial_lr: )[0-9.eE-]+",
                         lambda mo: mo.group(1) + f"{presets['resnet50']:g}", new, count=1)
        if k != 1:
            F.warn("could not set the starting learning rate in annotation.jsx: "
                   "set lrConfig.initial_lr to the ResNet-50 preset by hand")
    if new != src:
        path.write_text(new)
        F.notes.append(f"annotation.jsx now has: {line} (commit it with the release)")
    else:
        F.notes.append(f"annotation.jsx already has: {line}")


def check_keys(F, tex_dir):
    tex = tex_dir / "SimplyAL_revised.tex"
    if not tex.exists():
        F.warn(f"{tex} not found: cannot check which values the paper still needs")
        return
    used = set(re.findall(r"\\res\{([^}]+)\}", tex.read_text()))
    manual = tex_dir / "results_manual.tex"
    hand = set(re.findall(r"\\setres\{([^}]+)\}", manual.read_text())) if manual.exists() else set()
    left = sorted(used - set(F.values) - hand)
    if left:
        F.warn("still missing, shown in red in the PDF: " + ", ".join(left))


def write_outputs(F, tex_dir, sources):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"% Written by fill_results.py on {stamp} from {', '.join(sources)}.",
             "% Rerunning fill_results.py overwrites this file; put hand-filled values in results_manual.tex."]
    lines += [f"\\setres{{{k}}}{{{v}}}" for k, v in sorted(F.values.items())]
    (tex_dir / "results.tex").write_text("\n".join(lines) + "\n")
    snippets = ["Replace each [[NAME]] in the response letter with the text below it.", ""]
    for name in SNIPPETS:
        snippets += [f"[[{name}]]", F.snippets.get(name, "(missing: see results_report.txt)"), ""]
    (tex_dir / "letter_snippets.txt").write_text("\n".join(snippets))
    report = [f"fill_results.py, {stamp}", "", "WARNINGS"]
    report += [f"- {w}" for w in F.warnings] or ["- none"]
    report += ["", "NOTES"] + ([f"- {n}" for n in F.notes] or ["- none"])
    report += ["", "SENTENCES WRITTEN INTO THE PAPER"]
    for key in ("efficiency-findings", "impact-findings", "sensitivity-findings", "guidance-lr", "abstract-random"):
        if F.values.get(key):
            report.append(f"{key}: {plain(F.values[key])}")
    report += ["", "ALL VALUES"] + [f"{k} = {v}" for k, v in sorted(F.values.items())]
    (tex_dir / "results_report.txt").write_text("\n".join(report) + "\n")


def main():
    args = parse_args()
    F = Filled()
    res, summ, plat, env = read_runs(args.runs)
    sources = [str(args.runs)]
    cpu_summ, cpu_env = None, {}
    if args.cpu_runs:
        if (args.cpu_runs / "summary.csv").exists():
            cpu_summ = pd.read_csv(args.cpu_runs / "summary.csv")
            if (args.cpu_runs / "env.json").exists():
                cpu_env = json.loads((args.cpu_runs / "env.json").read_text())
            sources.append(str(args.cpu_runs))
        else:
            F.warn(f"{args.cpu_runs} has no summary.csv")
    if "selection_error" in res.columns:
        errors = int(res.selection_error.astype(str).str.lower().eq("true").sum())
        if errors:
            F.warn(f"{errors} episodes logged a selection error (that batch fell back to random); "
                   "see runs/run_all.log")
    fill_env(F, env)
    fill_sizes(F, res, env)
    p_lc = fill_efficiency(F, res, plat)
    at50 = fill_sensitivity(F, summ)
    presets = fill_presets(F, at50)
    fill_latency(F, summ, cpu_summ, cpu_env)
    fill_data(F, args)
    (args.tex_dir / "figures").mkdir(parents=True, exist_ok=True)
    if not matches(res, None, "least_confidence").empty:
        plot_figure5(res, p_lc, args.tex_dir / "figures" / "active_learning_curves.pdf")
        F.notes.append("Figure 5 written to figures/active_learning_curves.pdf")
    if args.repo:
        write_presets(F, args.repo, presets)
    else:
        F.notes.append("no --repo: learning-rate presets not written; values: "
                       + ", ".join(f"{m} {presets.get(m, 1e-3):g}" for m in MODELS))
    check_keys(F, args.tex_dir)
    write_outputs(F, args.tex_dir, sources)
    print(f"Wrote {len(F.values)} values to {args.tex_dir / 'results.tex'}")
    print(f"Report: {args.tex_dir / 'results_report.txt'}")
    print(f"Letter text: {args.tex_dir / 'letter_snippets.txt'}")
    for note in F.notes:
        print("NOTE:", note)
    for warning in F.warnings:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()
