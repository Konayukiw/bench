from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
BUILD = DATA / "build"
MODEL_LIST = ROOT / "known_llm_models.md"
RELEASE_DATES = DATA / "release_dates.csv"
ALIASES = DATA / "aliases.csv"
MANUAL = DATA / "manual" / "observations.csv"
LEGACY = DATA / "legacy" / "known-llm-benchmarks.2026-09-05.json"
PREVIOUS_CSV = DATA / "legacy" / "unified_scores.2026-09-12.csv"

OUT_CSV = ROOT / "unified_scores.csv"
OUT_EST_CSV = ROOT / "unified_scores_estimated.csv"
OUT_JSON = ROOT / "unified_scores.json"
OUT_SUMMARY = ROOT / "unified_scores_summary.json"

P_STANDARD = 0
P_LEADERBOARD = 1
P_UNKNOWN = 2
P_SELF = 3
GROUP_OF_PRIORITY = {P_STANDARD: "standard", P_LEADERBOARD: "leaderboard", P_UNKNOWN: "unknown", P_SELF: "self"}
GROUP_WEIGHT = {"standard": 1.0, "leaderboard": 0.85, "unknown": 0.7, "self": 0.5}
MEASURED_GROUPS = ("standard", "leaderboard")

FAMILIES = ["static_code", "agentic_swe", "terminal", "security"]

BENCHMARKS = {
    "SWE-bench Verified": {"family": "agentic_swe", "anchor": "SWE-Bench verified"},
    "Aider Polyglot": {"family": "static_code", "anchor": "Aider polyglot"},
    "Terminal-Bench 2.0": {"family": "terminal", "anchor": "Terminal Bench"},
    "DeepSWE": {"family": "agentic_swe", "anchor": "DeepSWE"},
    "WeirdML": {"family": "static_code", "anchor": "WeirdML"},
    "Cybench": {"family": "security", "anchor": "Cybench", "role": "subscore"},
    "GSO": {"family": "agentic_swe", "anchor": "GSO-Bench"},
    "FrontierCode": {"family": "agentic_swe", "anchor": "FrontierCode"},
    "FrontierSWE": {"family": "agentic_swe", "anchor": "FrontierSWE"},
    "MirrorCode": {"family": "agentic_swe", "anchor": "MirrorCode"},
    "PostTrainBench": {"family": "agentic_swe", "anchor": "PostTrainBench"},
    "Surface Evolver Bench": {"family": "static_code", "anchor": "Surface Evolver Bench"},
    "ExploitBench": {"family": "security", "anchor": "ExploitBench", "role": "subscore"},
    "WeirdML v3": {"family": "static_code"},
    "CursorBench": {"family": "agentic_swe"},
    "SciCode": {"family": "static_code"},
    "LiveBench Coding": {"family": "static_code"},
    "LiveCodeBench": {"family": "static_code"},
    "LiveCodeBench (official)": {"family": "static_code"},
    "HumanEval": {"family": "static_code"},
    "HumanEval+": {"family": "static_code"},
    "MBPP": {"family": "static_code"},
    "MBPP+": {"family": "static_code"},
    "BigCodeBench": {"family": "static_code"},
    "BigCodeBench-Instruct": {"family": "static_code"},
    "OJBench": {"family": "static_code"},
    "DS-1000": {"family": "static_code"},
    "KernelBench": {"family": "static_code"},
    "SWE-bench Lite": {"family": "agentic_swe"},
    "SWE-bench Multilingual": {"family": "agentic_swe"},
    "SWE-bench Multimodal": {"family": "agentic_swe"},
    "SWE-bench Pro": {"family": "agentic_swe"},
    "SWE-bench Pro (Scale; mini-swe-agent)": {"family": "agentic_swe"},
    "SWE-bench Pro (Scale; SWE-Agent)": {"family": "agentic_swe"},
    "ProgramBench": {"family": "agentic_swe"},
    "Fullstack Bench": {"family": "agentic_swe"},
    "Vibe Code Bench": {"family": "agentic_swe"},
    "Terminal-Bench 2.1": {"family": "terminal"},
    "Terminal-Bench 4.0": {"family": "terminal"},
    "Terminal-Bench Hard": {"family": "terminal"},
    "Terminal-Bench 4.0 (Vals)": {"family": "terminal"},
    "Terminal-Bench 2.1 (Vals)": {"family": "terminal"},
    "Terminal-Bench 2.0 (Vals)": {"family": "terminal"},
    "SWE-bench Verified (Vals)": {"family": "agentic_swe"},
    "LiveCodeBench (Vals)": {"family": "static_code"},
    "IOI (Vals)": {"family": "static_code"},
    "Vibe Code Bench (Vals)": {"family": "agentic_swe"},
    "Vibe Code Bench 1-100 (Vals)": {"family": "agentic_swe"},
    "ProgramBench (Vals)": {"family": "agentic_swe"},
    "Code Migration (Vals)": {"family": "agentic_swe"},
    "SkillsBench (Vals)": {"family": "agentic_swe", "role": "validate"},
    "Arena WebDev Elo": {"family": "webdev", "kind": "elo", "role": "validate"},
    "AA Coding Index": {"family": "composite", "role": "validate"},
}

DYNAMIC_BENCHMARKS = {
    "SWE-rebench ": {"family": "agentic_swe", "share_group": "SWE-rebench"},
}


def spec(bench):
    if bench in BENCHMARKS:
        return BENCHMARKS[bench]
    for prefix, s in DYNAMIC_BENCHMARKS.items():
        if bench.startswith(prefix):
            return s
    raise KeyError(f"unknown benchmark key: {bench}")


VALS_BENCHMARKS = {
    "terminal-bench-4": "Terminal-Bench 4.0 (Vals)",
    "terminal-bench-2-1": "Terminal-Bench 2.1 (Vals)",
    "terminal-bench-2": "Terminal-Bench 2.0 (Vals)",
    "swebench": "SWE-bench Verified (Vals)",
    "lcb": "LiveCodeBench (Vals)",
    "ioi": "IOI (Vals)",
    "vibe-code": "Vibe Code Bench (Vals)",
    "vcb-1-100": "Vibe Code Bench 1-100 (Vals)",
    "programbench": "ProgramBench (Vals)",
    "code-migration": "Code Migration (Vals)",
    "skillsbench": "SkillsBench (Vals)",
}

LEGACY_BENCH_MAP = {name: name for name in BENCHMARKS if "(" not in name}
LEGACY_BENCH_MAP.pop("AA Coding Index")
LEGACY_SKIP_SOURCES = ("vals.ai", "swe-rebench.com")

MIN_MODELS_PER_BENCH = 5
OFFSET_SHRINK_K = 5.0
BOOTSTRAP = 200
SEED = 7

SCORE_SLOPE = 50.0
SCORE_INTERCEPT = -2500.0
LOW_INFO_CI_WIDTH = 1500.0

SELF_SOURCE_PATTERN = r"system card|model card|blog|announc|technical report|tech report|press release|launch|self-reported|self reported|official (?:chart|table|card)"

VENDOR_DOMAINS = {
    "OpenAI": ["openai.com"],
    "Anthropic": ["anthropic.com", "claude.com"],
    "Google": ["deepmind.google", "blog.google", "ai.google.dev", "storage.googleapis.com/deepmind"],
    "Meta": ["ai.meta.com", "llama.com"],
    "xAI": ["x.ai"],
    "DeepSeek": ["deepseek.com", "huggingface.co/deepseek-ai"],
    "Alibaba": ["qwenlm.github.io", "qwen.ai", "huggingface.co/qwen", "alibabacloud.com"],
    "Moonshot AI": ["moonshot.ai", "moonshot.cn", "kimi.com", "huggingface.co/moonshotai"],
    "Zhipu AI (Z.ai)": ["z.ai", "zhipuai.cn", "huggingface.co/zai-org"],
    "Mistral AI": ["mistral.ai"],
    "Anysphere": ["cursor.com"],
    "Cognition": ["cognition.com", "cognition.ai", "windsurf.com"],
    "Poolside": ["poolside.ai"],
    "Xiaomi": ["mimo.xiaomi.com", "huggingface.co/xiaomimimo"],
    "MiniMax": ["minimax.io", "minimaxi.com"],
}
