import numpy as np
import scipy.stats as stats


def compute_metrics(y_true, y_pred, label=""):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pearson_r,  pearson_p  = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    print(f"\n{'─'*48}")
    if label:
        print(f"  {label}")
    print(f"  N        : {mask.sum()}")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  Pearson  : r={pearson_r:.4f}  p={pearson_p:.3g}")
    print(f"  Spearman : r={spearman_r:.4f}  p={spearman_p:.3g}")
    print(f"{'─'*48}\n")
    return dict(n=int(mask.sum()), rmse=rmse,
                pearson_r=pearson_r,  pearson_p=pearson_p,
                spearman_r=spearman_r, spearman_p=spearman_p)


def split_random_mask(long_df, holdout_frac=0.20, random_state=42):
    rand_num_gen  = np.random.default_rng(random_state)
    n        = len(long_df)
    test_idx = rand_num_gen.choice(n, size=int(n * holdout_frac), replace=False)
    mask     = np.zeros(n, dtype=bool)
    mask[test_idx] = True

    train_df = long_df[~mask].copy()
    test_df  = long_df[ mask].copy()

    print(f"[random mask]  train obs={len(train_df)}  test obs={len(test_df)}")
    return train_df, test_df


def split_holdout_peptides(long_df, holdout_frac=0.20, random_state=42):
    #set holdouts
    peptides  = long_df["Peptide ID"].unique()
    rand_num_gen = np.random.default_rng(random_state)
    n_holdout = max(1, int(len(peptides) * holdout_frac))
    held_out  = set(rand_num_gen.choice(peptides, size=n_holdout, replace=False))

    train_df = long_df[~long_df["Peptide ID"].isin(held_out)].copy()
    test_df  = long_df[ long_df["Peptide ID"].isin(held_out)].copy()

    print(f"[peptide holdout]  train peptides={train_df['Peptide ID'].nunique()}"
          f"  test peptides={test_df['Peptide ID'].nunique()}  ({len(test_df)} obs)")
    return train_df, test_df