import pandas as pd

DS = ["cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than"]


def probe_train_statements(data, n=400, seed=0):
    """The exact rows e00 probes were trained on (same sampling as experiments/e00_baselines/probes.py)."""
    return set(pd.concat([pd.read_csv(f"{data}/{d}.csv").sample(frac=1, random_state=seed).groupby("label").head(n // 2)
                          for d in DS]).statement)


def heldout_generic(data, n, seed):
    """n generic statements (equal per dataset) that the e00 probes never saw."""
    seen = probe_train_statements(data)
    out = []
    for d in DS:
        df = pd.read_csv(f"{data}/{d}.csv")
        df = df[~df.statement.isin(seen)]
        out.append(df.sample(min(n // len(DS), len(df)), random_state=seed))
    return pd.concat(out)
