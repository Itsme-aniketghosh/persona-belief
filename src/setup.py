"""One config per prior-paper setup: claims, the three conditions (none / persona / control), labels, cells.
Condition keys stay "none", "hist", "ctrl" (persona, control) so earlier E12 results keep their field names."""
import csv, json, os

import pandas as pd

ROOT = os.path.expanduser("~/pb")
SUF = "Consider this statement: "  # probe format: statement read at its last token
ASK = "Is the following statement true or false? Answer with exactly one word, True or False.\nStatement: "


def load(name):
    cfg = json.load(open(f"{ROOT}/data/{name}/setup.json"))
    src = cfg.get("claims_from", name)
    cl = pd.DataFrame(list(csv.DictReader(open(f"{ROOT}/data/{src}/claims.csv", encoding="utf-8"))))
    return cfg, cl


def formats(tok, sys_prompts, ask_text=ASK):
    """wrap(cond, s): chat text ending at the statement (probe format). ask(cond, s): yes/no question with generation
    prompt (behavior format). span(text, piece): (tokens after piece, tokens up to piece end) counted from the end."""
    mark = "\x00"
    try:  # some templates (Gemma-2) reject a system role: then the persona text opens the user turn
        tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], tokenize=False)
        has_sys = True
    except Exception:
        has_sys = False

    def msgs(cond, content):
        sp = sys_prompts.get(cond)
        if sp and not has_sys:
            return [{"role": "user", "content": sp + "\n\n" + content}]
        return ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": content}]

    def wrap(cond, s):
        return tok.apply_chat_template(msgs(cond, mark), tokenize=False).split(mark)[0] + SUF + s

    def ask(cond, s):
        return tok.apply_chat_template(msgs(cond, ask_text + s), tokenize=False, add_generation_prompt=True)

    def span(text, piece):
        start = text.rindex(piece)
        off = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        idx = [i for i, (a, b) in enumerate(off) if a >= start and b <= start + len(piece) and b > a]
        n = len(off)
        return n - 1 - idx[-1], n - idx[0]  # (#tokens after the span, #tokens from span start to end)

    return wrap, ask, span


def file_tag(name):
    return "" if name == "historical" else f"{name}_"  # historical keeps its original E12 file names
