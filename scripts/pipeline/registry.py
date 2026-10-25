import csv
import re
import unicodedata
from collections import defaultdict

from . import config

VARIANT_TOKENS = {"instruct", "chat", "it", "hf", "latest", "exp", "experimental", "preview", "fp8", "bf16", "api", "beta", "release", "version"}
SOURCE_ONLY_TOKENS = {"thinking", "reasoning", "nonreasoning", "non", "no", "think", "nothink"}
BRAND_TOKENS = {"openai", "anthropic", "google", "meta", "microsoft", "amazon", "ibm", "tencent", "baidu", "xai", "alibaba", "mistralai", "nvidia", "cohere", "bytedance", "moonshotai", "zai", "deepseekai", "accounts", "fireworks", "models", "hfendpoint", "together", "bedrock", "vertex", "azure", "openrouter"}
WARNING_SIGN = chr(0x26A0) + chr(0xFE0F)
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december"


def clean(s):
    s = unicodedata.normalize("NFKC", str(s)).replace(chr(0x200B), "")
    return re.sub(r"\s+", " ", s).strip()


def key_exact(s):
    return re.sub(r"[^a-z0-9+]", "", clean(s).lower())


def tokens(s):
    s = clean(s).lower()
    s = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", s)
    s = re.sub(rf"\b(?:{MONTHS})\.? ?\d{{4}}\b", " ", s)
    s = re.sub(r"\b20\d{2}[-_/]?\d{2}[-_/]?\d{2}\b", " ", s)
    s = re.sub(r"\b\d{8}\b", " ", s)
    s = re.sub(r"\b\d{2}-(?:20)?\d{2}\b(?!\.)", " ", s)
    s = re.sub(r"[^a-z0-9.+]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    out = []
    for t in s.split():
        if re.fullmatch(r"a\d+(?:\.\d+)?b|\d+x\d+b", t):
            out.append(t)
        else:
            out.extend(x for x in re.split(r"(?<=[a-z])(?=\d)", t) if x)
    return out


def core_tokens(s, source):
    drop = VARIANT_TOKENS | BRAND_TOKENS | (SOURCE_ONLY_TOKENS if source else set())
    return [x for x in tokens(s) if x not in drop]


def key_strict(s, source=False):
    drop = BRAND_TOKENS | (SOURCE_ONLY_TOKENS if source else set())
    return "".join(x for x in tokens(s) if x not in drop).replace(".", "")


def key_loose(s, source=False):
    return "".join(core_tokens(s, source)).replace(".", "")


def key_nosize(s, source=False):
    t = [x for x in core_tokens(s, source) if not re.fullmatch(r"a?\d+(?:\.\d+)?[bmk]|\d+x\d+b|\d+e", x)]
    return "".join(t).replace(".", "")


def key_sorted(s, source=False):
    return " ".join(sorted(y for x in core_tokens(s, source) for y in x.split(".") if y))


def parse_model_list(path=config.MODEL_LIST):
    models = []
    vendor = series = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            vendor, series = clean(line[3:]), None
        elif line.startswith("### "):
            series = clean(line[4:])
        elif line.startswith("- ") and vendor:
            name = clean(re.sub(rf"[\s{WARNING_SIGN}]+$", "", line[2:]))
            if name and not name.startswith("**"):
                models.append({"model": name, "vendor": vendor, "series": series or vendor, "order": len(models)})
    return models


def load_release_dates(path=config.RELEASE_DATES):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {clean(r["model"]): r["release_date"].strip() for r in csv.DictReader(f) if r["release_date"].strip()}


def load_aliases(path=config.ALIASES):
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["source"].strip() or "*", key_exact(r["name"]))] = clean(r["model"])
    return out


class Matcher:
    def __init__(self, models, aliases):
        self.models = {m["model"]: m for m in models}
        self.aliases = aliases
        self.index = {}
        for fn in (key_exact, key_loose, key_nosize, key_sorted):
            buckets = defaultdict(set)
            for name in self.models:
                buckets[fn(name)].add(name)
            self.index[fn.__name__] = buckets
        self.log = []

    def match(self, source, name):
        k = key_exact(name)
        for src in (source, "*"):
            if (src, k) in self.aliases:
                target = self.aliases[(src, k)]
                if target == "-":
                    return None, "ignored"
                if target not in self.models:
                    raise ValueError(f"alias target not in model list: {target}")
                return target, "alias"
        stages = [(key_exact, None)] + [(fn, s) for fn in (key_loose, key_nosize, key_sorted) for s in (False, True)]
        for fn, src_strip in stages:
            k = fn(name) if src_strip is None else fn(name, source=src_strip)
            if not k:
                continue
            hits = self.index[fn.__name__].get(k, set())
            if len(hits) == 1:
                target = next(iter(hits))
                if fn is not key_exact:
                    self.log.append({"source": source, "name": name, "model": target, "stage": fn.__name__})
                return target, fn.__name__
            if len(hits) > 1:
                exact = [h for h in hits if key_exact(h) == key_exact(name)]
                if len(exact) == 1:
                    return exact[0], fn.__name__
                strict = [h for h in hits if key_strict(h) == key_strict(name, source=True)]
                if len(strict) == 1:
                    self.log.append({"source": source, "name": name, "model": strict[0], "stage": fn.__name__ + "+strict"})
                    return strict[0], fn.__name__
                self.log.append({"source": source, "name": name, "model": " | ".join(sorted(hits)), "stage": "ambiguous:" + fn.__name__})
                return None, "ambiguous"
        self.log.append({"source": source, "name": name, "model": "", "stage": "unmatched"})
        return None, "unmatched"
