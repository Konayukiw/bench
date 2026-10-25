import csv
import datetime as dt
import json
import re
from collections import defaultdict

import pandas as pd
import yaml

from . import config
from .config import P_LEADERBOARD, P_SELF, P_STANDARD, P_UNKNOWN

EFFORT_SUFFIX = re.compile(r"_(max|xhigh|high|medium|low|minimal|none|unknown|default|thinking|nothinking)$")
SELF_RE = re.compile(config.SELF_SOURCE_PATTERN, re.I)

EPOCH_FILES = [
    ("swe_bench_verified.csv", "SWE-bench Verified", "Best score (across scorers)", P_STANDARD),
    ("mirrorcode.csv", "MirrorCode", "Best score (across scorers)", P_STANDARD),
    ("aider_polyglot_external.csv", "Aider Polyglot", "Percent correct", P_STANDARD),
    ("terminalbench_external.csv", "Terminal-Bench 2.0", "Accuracy mean", P_LEADERBOARD),
    ("deepswe_external.csv", "DeepSWE", "Pass@1", P_STANDARD),
    ("weirdml_external.csv", "WeirdML", "Accuracy", P_STANDARD),
    ("weirdml_v3_external.csv", "WeirdML v3", "Score", P_STANDARD),
    ("gso_external.csv", "GSO", "Score OPT@1", P_STANDARD),
    ("frontierswe_external.csv", "FrontierSWE", "Score", P_STANDARD),
    ("frontiercode_external.csv", "FrontierCode", "Main score", P_STANDARD),
    ("cybench_external.csv", "Cybench", "Unguided % Solved", P_STANDARD),
    ("exploitbench_external.csv", "ExploitBench", "Mean capability", P_STANDARD),
    ("posttrainbench_external.csv", "PostTrainBench", "Average (%)", P_STANDARD),
    ("surface_evolver_bench_external.csv", "Surface Evolver Bench", "Mean score", P_STANDARD),
    ("cursorbench_external.csv", "CursorBench", "Score", P_STANDARD),
    ("scicode_external.csv", "SciCode", "Score", P_STANDARD),
    ("live_bench_external.csv", "LiveBench Coding", "Coding average", P_STANDARD),
    ("webdev_arena_external.csv", "Arena WebDev Elo", "Arena Score", P_STANDARD),
]
SE_COLUMNS = [("stderr", 1.0), ("Accuracy SE", 1.0), ("95% CI half-width", 1 / 1.96)]
CONFIG_COLUMNS = ["Agent", "Harness", "Scaffold", "Reasoning effort", "Reasoning level", "Provider", "Edit format"]


def latest(source):
    dirs = sorted(p for p in (config.RAW / source).glob("????-??-??") if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"no snapshot for {source} under {config.RAW / source}")
    return dirs[-1]


def obs(source, name, bench, value, priority, cfg="", url="", date="", note="", source_text="", se=None):
    return {"source": source, "name": str(name).strip(), "bench": bench, "value": float(value), "priority": priority, "se": None if se is None else float(se),
            "config": str(cfg or "").strip(), "url": str(url or "").strip(), "date": str(date or "").strip(),
            "note": str(note or "").strip(), "source_text": str(source_text or "").strip()}


def epoch_se(r, scale):
    for col, k in SE_COLUMNS:
        v = r.get(col)
        if v is not None and pd.notna(v):
            return float(v) * k * scale
    hi, lo = r.get("95% CI High"), r.get("95% CI Low")
    if hi is not None and lo is not None and pd.notna(hi) and pd.notna(lo):
        return (float(hi) - float(lo)) / (2 * 1.96)
    return None


def to_frac(values):
    return 0.01 if values.max() > 1.5 else 1.0


def load_epoch():
    snap = latest("epoch")
    base = snap / "benchmark_data"
    meta = pd.read_csv(base / "model_metadata.csv").dropna(subset=["model_version"])
    group = dict(zip(meta.model_version, meta.model_group))
    out = []
    for fname, bench, col, prio in EPOCH_FILES:
        df = pd.read_csv(base / fname)
        df = df[pd.to_numeric(df[col], errors="coerce").notna()].copy()
        df[col] = df[col].astype(float)
        elo = config.spec(bench).get("kind") == "elo"
        scale = 1.0 if elo else to_frac(df[col])
        for _, r in df.iterrows():
            version = r["Model version"]
            if pd.isna(version):
                if "Name" not in df.columns or pd.isna(r.get("Name")):
                    continue
                version = str(r["Name"])
            version = str(version)
            name = group.get(version)
            if not isinstance(name, str) or not name:
                name = EFFORT_SUFFIX.sub("", version)
            effort = EFFORT_SUFFIX.search(version)
            cfg = [effort.group(1)] if effort else []
            cfg += [f"{c}={r[c]}" for c in CONFIG_COLUMNS if c in df.columns and pd.notna(r.get(c))]
            text = " ".join(str(r[c]) for c in ("Source", "Source link", "Notes") if c in df.columns and pd.notna(r.get(c)))
            p = P_SELF if SELF_RE.search(text) else prio
            url = r.get("Source") if isinstance(r.get("Source"), str) and r.get("Source", "").startswith("http") else "https://epoch.ai/benchmarks"
            out.append(obs("epoch", name, bench, r[col] * scale, p, "; ".join(cfg), url, snap.name, f"epoch:{version}", text, epoch_se(r, scale)))
    return out


def load_epoch_eci():
    snap = latest("epoch")
    eci = pd.read_csv(snap / "eci_scores.csv")
    edi = pd.read_csv(snap / "edi_scores.csv")
    processed = pd.read_csv(snap / "eci_benchmarks.csv")
    return eci, edi, processed


def load_aa():
    try:
        snap = latest("aa")
    except FileNotFoundError:
        return []
    d = json.loads((snap / "aa_models.json").read_text(encoding="utf-8"))
    out = []
    for m in d["data"]:
        v = (m.get("evaluations") or {}).get("artificial_analysis_coding_index")
        if v is None:
            continue
        name = m["name"]
        cfg = ", ".join(re.findall(r"\(([^()]*)\)", name))
        out.append(obs("aa", name, "AA Coding Index", v / 100.0, P_STANDARD, cfg,
                       "https://artificialanalysis.ai/", snap.name, f"aa:{m.get('slug')}"))
    return out


def load_swebench():
    snap = latest("swebench")
    d = json.loads((snap / "leaderboards.json").read_text(encoding="utf-8"))
    boards = {"Verified": "SWE-bench Verified", "Lite": "SWE-bench Lite", "Multimodal": "SWE-bench Multimodal", "Multilingual": "SWE-bench Multilingual"}
    out = []
    for b in d["leaderboards"]:
        bench = boards.get(b.get("name"))
        if not bench:
            continue
        for r in b.get("results", []):
            if r.get("resolved") is None:
                continue
            tags = r.get("tags") or []
            if isinstance(tags, str):
                tags = re.findall(r"'([^']*)'", tags)
            models = [t.split(":", 1)[1].strip() for t in tags if t.startswith("Model:")]
            if len(models) > 1:
                continue
            name = r.get("model_display") or (models[0] if models else r.get("name"))
            agent = r.get("agent") or ""
            prio = P_STANDARD if "mini-swe-agent" in agent.lower().replace(" ", "-") else P_LEADERBOARD
            if not r.get("checked"):
                prio = P_UNKNOWN
            cfg = "; ".join(x for x in (agent, f"effort={r.get('reasoning_effort')}" if r.get("reasoning_effort") else "") if x)
            out.append(obs("swebench", name, bench, float(r["resolved"]) / 100.0, prio, cfg, "https://www.swebench.com/", r.get("date", ""), r.get("folder", "")))
    return out


def load_evalplus():
    snap = latest("evalplus")
    d = json.loads((snap / "results.json").read_text(encoding="utf-8"))
    keys = {"humaneval": "HumanEval", "humaneval+": "HumanEval+", "mbpp": "MBPP", "mbpp+": "MBPP+"}
    out = []
    for name, r in d.items():
        for k, bench in keys.items():
            v = (r.get("pass@1") or {}).get(k)
            if v is not None:
                out.append(obs("evalplus", name, bench, float(v) / 100.0, P_STANDARD, "", "https://evalplus.github.io/leaderboard.html", snap.name))
    return out


def load_bigcodebench():
    snap = latest("bigcodebench")
    d = json.loads((snap / "results.json").read_text(encoding="utf-8"))
    out = []
    for name, r in d.items():
        for split, bench in (("complete", "BigCodeBench"), ("instruct", "BigCodeBench-Instruct")):
            v = (r.get("pass@1") or {}).get(split)
            if v is not None:
                out.append(obs("bigcodebench", name, bench, float(v) / 100.0, P_STANDARD, split, "https://bigcode-bench.github.io/", r.get("date", "")))
    return out


def next_flight_payload(html):
    chunks = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html)
    return "".join(json.loads(c) for c in chunks)


def find_holder(o, key, depth=0):
    if depth > 14:
        return None
    if isinstance(o, dict):
        if key in o:
            return o
        children = o.values()
    elif isinstance(o, list):
        children = o
    else:
        return None
    for v in children:
        r = find_holder(v, key, depth + 1)
        if r is not None:
            return r
    return None


_SWEREBENCH_CACHE = {}


def swerebench_items():
    if "data" not in _SWEREBENCH_CACHE:
        try:
            snap = latest("swerebench")
        except FileNotFoundError:
            _SWEREBENCH_CACHE["data"] = (None, [], [])
            return _SWEREBENCH_CACHE["data"]
        payload = next_flight_payload((snap / "index.html").read_text(encoding="utf-8"))
        line = next(l for l in payload.split("\n") if '"rangeStats"' in l and '"items"' in l)
        holder = find_holder(json.loads(line.split(":", 1)[1]), "items")
        dates = holder["dates"]["dates"] if isinstance(holder.get("dates"), dict) else holder["dates"]
        items = [it for it in holder["items"] if (it.get("meta") or {}).get("instance_type") == "model"]
        _SWEREBENCH_CACHE["data"] = (snap.name, dates, items)
    return _SWEREBENCH_CACHE["data"]


def swerebench_name(raw):
    name = re.sub(r"\s*\[[^\]]*\]\s*$", "", raw)
    return re.sub(r"-(minimal|low|medium|high|xhigh|max)$", "", name)


def load_swerebench():
    snap_name, dates, items = swerebench_items()
    if not items:
        return []
    month_starts = [t for t in dates if dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).day == 1]
    out = []
    for it in items:
        lo, hi = it["taskRangeTimestamp"]["from"], it["taskRangeTimestamp"]["to"]
        name = swerebench_name(it["modelName"])
        effort = re.search(r"\[([^\]]*)\]|-(minimal|low|medium|high|xhigh|max)$", it["modelName"])
        for a, b in zip(month_starts, month_starts[1:]):
            if lo > a or hi < b:
                continue
            s = it["rangeStats"]["all"].get(f"{a}:{b}")
            if not s or s.get("resolvedRate") is None:
                continue
            month = dt.datetime.fromtimestamp(a / 1000, dt.timezone.utc).strftime("%Y-%m")
            cfg = f"agent={it.get('agentVersion')}" + (f"; effort={effort.group(1) or effort.group(2)}" if effort else "")
            out.append(obs("swerebench", name, f"SWE-rebench [{it.get('agentVersion')}] {month}", s["resolvedRate"] / 100.0, P_STANDARD, cfg,
                           "https://swe-rebench.com/", snap_name, it.get("modelId", ""), se=None if s.get("sem") is None else s["sem"] / 100.0))
    return out


def load_vals():
    try:
        snap = latest("vals")
    except FileNotFoundError:
        return []
    out = []
    for slug, bench in config.VALS_BENCHMARKS.items():
        path = snap / f"{slug}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        meta = d.get("metadata", {})
        for key, v in (d.get("tasks", {}).get("overall") or {}).items():
            if v.get("accuracy") is None:
                continue
            effort = v.get("reasoning_effort") or v.get("compute_effort")
            harness = v.get("harness") or meta.get("harness")
            cfg = "; ".join(x for x in (f"effort={effort}" if effort else "", f"harness={harness}" if harness else "") if x)
            out.append(obs("vals", key.split("/", 1)[-1], bench, float(v["accuracy"]) / 100.0, P_STANDARD, cfg,
                           f"https://www.vals.ai/benchmarks/{slug}", meta.get("updated", snap.name), key, se=None if v.get("stderr") is None else float(v["stderr"]) / 100.0))
    return out


def load_livecodebench(start="2024-08-01", min_problems=100):
    snap = latest("livecodebench")
    d = json.loads((snap / "performances_generation.json").read_text(encoding="utf-8"))
    t0 = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp() * 1000
    acc = defaultdict(list)
    for p in d["performances"]:
        if p.get("date", 0) >= t0 and p.get("pass@1") is not None:
            acc[p["model"]].append(float(p["pass@1"]))
    out = []
    for name, v in acc.items():
        if len(v) >= min_problems:
            out.append(obs("livecodebench", name, "LiveCodeBench (official)", sum(v) / len(v) / 100.0, P_STANDARD,
                           f"problems since {start} (n={len(v)})", "https://livecodebench.github.io/leaderboard.html", snap.name))
    return out


def load_aider():
    snap = latest("aider")
    rows = yaml.safe_load((snap / "polyglot_leaderboard.yml").read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if r.get("pass_rate_2") is None:
            continue
        name = r["model"]
        cfg = "; ".join(x for x in (r.get("edit_format"), f"effort={r['reasoning_effort']}" if r.get("reasoning_effort") else "") if x)
        out.append(obs("aider", name, "Aider Polyglot", float(r["pass_rate_2"]) / 100.0, P_STANDARD, cfg,
                       "https://aider.chat/docs/leaderboards/", str(r.get("date", ""))))
    return out


def is_base_checkpoint(model, setting):
    if "base" in model.lower():
        return False
    main = setting.lower().split("(")[0].split(";")[0]
    if "post-trained" in main or "instruct" in main:
        return False
    return bool(re.search(r"\bbase\b", main))


def legacy_key(bench, setting):
    if "evalplus (base)" in setting.lower():
        return {"MBPP+": "MBPP", "HumanEval+": "HumanEval"}.get(bench, bench)
    return config.LEGACY_BENCH_MAP.get(bench)


def load_legacy():
    if not config.LEGACY.exists():
        return []
    d = json.loads(config.LEGACY.read_text(encoding="utf-8"))
    out = []
    for m in d["models"]:
        for section in ("scores", "self_reported"):
            for bench, e in (m.get(section) or {}).items():
                v = e.get("value")
                setting = e.get("setting") or ""
                key = legacy_key(bench, setting)
                if bench.startswith("BigCodeBench") or is_base_checkpoint(m["name"], setting):
                    continue
                if not key or not isinstance(v, (int, float)) or any(s in (e.get("source") or "") for s in config.LEGACY_SKIP_SOURCES):
                    continue
                prov = e.get("provenance") or ("self_reported" if section == "self_reported" else "unknown")
                if prov == "measured":
                    prio = P_STANDARD if "AA harness" in setting else P_LEADERBOARD
                elif prov == "self_reported":
                    prio = P_SELF
                else:
                    prio = P_UNKNOWN
                unit = e.get("unit")
                value = float(v) if unit == "elo" else float(v) / 100.0
                if unit not in ("elo",) and value > 1.0001:
                    continue
                out.append(obs("legacy", m["name"], key, value, prio, setting, e.get("source", ""), e.get("source_date", ""), "legacy:2026-09-05", setting))
    return out


def load_manual():
    if not config.MANUAL.exists():
        return []
    prio = {"standard": P_STANDARD, "leaderboard": P_LEADERBOARD, "unknown": P_UNKNOWN, "self": P_SELF}
    out = []
    with config.MANUAL.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["model"].strip() or r["model"].startswith("#"):
                continue
            unit = r["unit"].strip()
            v = float(r["value"])
            div = 100.0 if unit == "percent" else 1.0
            ci = re.search(r"\xb1\s*([0-9.]+)", r.get("note", ""))
            out.append(obs("manual", r["model"], r["benchmark"].strip(), v / div, prio[r["provenance"].strip()], r.get("config", ""),
                           r.get("source_url", ""), r.get("retrieved", ""), r.get("note", ""), se=float(ci.group(1)) / 1.96 / div if ci else None))
    return out


LOADERS = [load_epoch, load_aa, load_swebench, load_swerebench, load_vals, load_evalplus, load_bigcodebench, load_livecodebench, load_aider, load_legacy, load_manual]


def load_all():
    out = []
    counts = {}
    for fn in LOADERS:
        rows = fn()
        counts[fn.__name__.removeprefix("load_")] = len(rows)
        out.extend(rows)
    for b in {o["bench"] for o in out}:
        config.spec(b)
    return out, counts
