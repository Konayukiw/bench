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
    # AA's own runs of FrontierCode. Not Epoch's anchored FrontierCode series:
    # v1.1 is a different harness, and 2.0 is a different task set (Intelligence Index v5).
    "FrontierCode 1.1 (AA)": {"family": "agentic_swe"},
    "FrontierCode 2.0 (AA)": {"family": "agentic_swe"},
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
    # v5 headline index is an order-only diagnostic. It is not a success probability
    # and is never a likelihood term. kind=index keeps it out of the logit offset fit
    # if a row is ever emitted.
    "AA Intelligence Index v5": {"family": "composite", "role": "validate", "kind": "index"},
}

# Public-site fraction fields -> benchmark key. Same-named keys join the existing
# series (one selected value per model). The FrontierCode keys stay unanchored.
AA_PUBLIC_FRACTIONS = {
    "livecodebench": "LiveCodeBench",
    "scicode": "SciCode",
    "terminalbenchHard": "Terminal-Bench Hard",
    "terminalBench21": "Terminal-Bench 2.1",
    "terminalBench40": "Terminal-Bench 4.0",
    "frontierCode": "FrontierCode 1.1 (AA)",
    "frontierCode20": "FrontierCode 2.0 (AA)",
}

# Point-estimate order the coding score is required to respect.
# hard: a reversal is an absurd collapse (different generation or size class).
# expected: adjacent flagships. Reported, and a clear reversal is called out,
# but a small overlap is not by itself a broken scale.
ESTABLISHED_PAIRS = [
    ("GPT-5.5", "GPT-4o", "hard", "OpenAI flagship two generations above GPT-4o"),
    ("GPT-5.5", "GPT-4", "hard", "OpenAI flagship above GPT-4"),
    ("GPT-5.5", "GPT-4o mini", "hard", "flagship above an older mini"),
    ("GPT-5.5", "GPT-3.5 Turbo", "hard", "flagship above GPT-3.5"),
    ("GPT-5", "GPT-4o", "hard", "GPT-5 above GPT-4o"),
    ("GPT-5", "GPT-4", "hard", "GPT-5 above GPT-4"),
    ("GPT-4o", "GPT-4", "hard", "GPT-4o above GPT-4"),
    ("GPT-4o", "GPT-4 Turbo", "hard", "GPT-4o above GPT-4 Turbo"),
    ("GPT-4o", "GPT-4o mini", "hard", "full model above its mini"),
    ("GPT-4o", "GPT-3.5 Turbo", "hard", "GPT-4o above GPT-3.5"),
    ("GPT-4", "GPT-3.5 Turbo", "hard", "GPT-4 above GPT-3.5"),
    ("OpenAI o3", "OpenAI o1", "hard", "o3 above o1"),
    ("OpenAI o3", "GPT-4o", "hard", "o3 above GPT-4o"),
    ("OpenAI o1", "GPT-4o", "hard", "o1 above GPT-4o"),
    ("OpenAI o1", "GPT-4", "hard", "o1 above GPT-4"),
    ("OpenAI o3", "OpenAI o3-mini", "hard", "o3 above o3-mini"),
    ("Claude Opus 4.5", "Claude Opus 4", "hard", "later Opus above Opus 4"),
    ("Claude Opus 4", "Claude 3.5 Sonnet", "hard", "Opus 4 above Claude 3.5 Sonnet"),
    ("Claude Opus 4", "Claude 3 Opus", "hard", "Opus 4 above Claude 3 Opus"),
    ("Claude 3.5 Sonnet", "Claude 3 Opus", "hard", "Claude 3.5 Sonnet above Claude 3 Opus"),
    ("Claude 3.5 Sonnet", "Claude 3 Haiku", "hard", "Claude 3.5 Sonnet above Claude 3 Haiku"),
    ("Claude Sonnet 4.5", "Claude 3.5 Sonnet", "hard", "Sonnet 4.5 above Claude 3.5 Sonnet"),
    ("Claude Opus 4.5", "Claude 3.5 Sonnet", "hard", "Opus 4.5 above Claude 3.5 Sonnet"),
    ("Claude Opus 4.5", "GPT-4o", "hard", "2025 Opus above GPT-4o"),
    ("Gemini 2.5 Pro", "Gemini 1.5 Pro", "hard", "Gemini 2.5 Pro above 1.5 Pro"),
    ("Gemini 2.5 Pro", "Gemini 1.5 Flash", "hard", "Gemini 2.5 Pro above 1.5 Flash"),
    ("Gemini 1.5 Pro", "Gemini 1.0 Pro", "hard", "Gemini 1.5 Pro above 1.0 Pro"),
    ("Gemini 2.5 Pro", "Gemini 2.0 Flash", "hard", "Gemini 2.5 Pro above 2.0 Flash"),
    ("Llama 3.1 405B", "Llama 3.1 70B", "hard", "405B above 70B"),
    ("Llama 3.1 70B", "Llama 3.1 8B", "hard", "70B above 8B"),
    ("Llama 3.1 405B", "Llama 2", "hard", "Llama 3.1 405B above Llama 2"),
    ("DeepSeek-V3", "DeepSeek-V2", "hard", "DeepSeek-V3 above V2"),
    ("DeepSeek-R1", "DeepSeek-V2", "hard", "DeepSeek-R1 above V2"),
    # Grok 2.5 has no observations. Its release date sits after Grok 4, so the
    # series interpolation is not a measurement that Grok 3 should outrank.
    ("Grok 4", "Grok 3", "hard", "Grok 4 above Grok 3"),
    ("Qwen3-Max", "Qwen2.5-Max", "hard", "Qwen3-Max above Qwen2.5-Max"),
    ("Qwen2.5-Max", "Qwen2.5", "hard", "Qwen2.5-Max above Qwen2.5"),
    ("GPT-4o", "Llama 3.1 8B", "hard", "GPT-4o above Llama 3.1 8B"),
    ("Claude 3.5 Sonnet", "Llama 3.1 8B", "hard", "Claude 3.5 Sonnet above Llama 3.1 8B"),
    ("GPT-5.5", "Claude 3.5 Sonnet", "hard", "GPT-5.5 above Claude 3.5 Sonnet"),
    ("Mistral Large 2.1", "Mistral Large 1.0", "hard", "Mistral Large 2.1 above Large 1.0"),
    ("GPT-6 Astra", "GPT-5.5", "expected", "adjacent OpenAI flagship"),
    ("Claude Opus 5.5", "Claude Opus 4.5", "expected", "adjacent Opus flagship"),
    ("Claude Opus 5", "Claude Opus 4.5", "expected", "Opus 5 above Opus 4.5"),
    ("Gemini 4 Argon", "Gemini 2.5 Pro", "expected", "Gemini 4 above Gemini 2.5 Pro"),
    ("Grok 4.5", "Grok 4", "expected", "adjacent Grok flagship"),
    ("DeepSeek-V4-Pro", "DeepSeek-V3", "expected", "DeepSeek-V4 Pro above V3"),
    ("Llama 4 Maverick", "Llama 3.1 70B", "expected", "Llama 4 Maverick above Llama 3.1 70B"),
    ("GPT-5.5", "OpenAI o3", "expected", "GPT-5.5 above o3"),
    ("Claude Opus 4.5", "Claude Sonnet 4.5", "expected", "same-generation Opus above Sonnet"),
    ("Gemini 3.1 Pro", "Gemini 2.5 Pro", "expected", "Gemini 3.1 Pro above 2.5 Pro"),
    ("DeepSeek-V3.2", "DeepSeek-V3", "expected", "DeepSeek-V3.2 above V3"),
    ("Mistral Large 4", "Mistral Large 2.1", "expected", "Mistral Large 4 above Large 2.1"),
]

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
