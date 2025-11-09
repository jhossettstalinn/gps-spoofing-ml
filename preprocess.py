import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def sampleDS(ds_original, RANDOM_STATE, N, e):
    """ Preprocess the dataset. """

    ds_noAttack = ds_original[ds_original['Output'] == 0]
    ds_attack = ds_original[ds_original['Output'] != 0]

    ds_noAttack_down = resample(
        ds_noAttack,
        n_samples=int(N*e),
        random_state=RANDOM_STATE,
        replace=False
        )

    ds_attack_down = resample(
        ds_attack,
        n_samples=int(N*(1-e)),
        random_state=RANDOM_STATE,
        replace=False
        )

    ds_balanced = pd.concat([ds_noAttack_down, ds_attack_down], 
                            ignore_index=True). \
                            sample(frac=1, random_state=RANDOM_STATE) \
                            .reset_index(drop=True)

    # binary class: 0=no_attack, {1,2,3}-->1=attack
    ds_sample = ds_balanced.copy()
    ds_sample['Output'] = (ds_sample['Output'] > 0).astype(int)

    #Save sampled dataset
    ds_sample.to_csv('outputs/ds_sampled.csv', index=False)
    print(f"Sampled dataset saved.\n")

    return ds_sample

def spearmancorr(X):
    """ Calculate Spearman correlation matrix """
    columns = X.columns
    X = X.values
    N, M = X.shape

    rank = np.apply_along_axis(stats.rankdata, axis=0, arr=X)
    S = np.zeros((M, M), dtype=float)

    denom = N * (N**2 - 1)
    for i in range(M):
        for j in range(M):
            d2 = np.sum((rank[:, i] - rank[:, j])**2)
            S[i, j] = (1 - (6 * d2) / denom).round(3)

    S = pd.DataFrame(S, index=columns, columns=columns)
    S.to_csv('outputs/spearman_correlation_matrix.csv')

    return S

def feature_to_drop(corr_mat, threshold):
    """ Select features to drop """
    to_drop = set()
    for i, feat_i in enumerate(corr_mat.columns):
        for j, feat_j in enumerate(corr_mat.columns):
            if i <= j:
                continue
            if abs(corr_mat.at[feat_i, feat_j]) > threshold:
                if feat_j != "DO":
                    to_drop.add(feat_j)
                else:
                    to_drop.add(feat_i)

    return to_drop


def lasso(X, y, RANDOM_STATE):

    C = 0.25
    max_iter=2000
    max_features=None

    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C,
        random_state=RANDOM_STATE,
        max_iter=max_iter
    )

    lasso.fit(X, y)

    abs_coef = np.abs(lasso.coef_).ravel()
    coef_X = pd.DataFrame({"feature": X.columns, "importance": abs_coef})
    coef_X = coef_X.sort_values("importance", ascending=False)

    if max_features is not None:
        sel_feat = coef_X.head(max_features)["feature"].tolist()
    else:
        sel_feat = coef_X.loc[coef_X["importance"] > 0, "feature"].tolist()

    to_drop = [c for c in X.columns if c not in sel_feat]

    coef_X.to_csv("outputs/Lasso_feature_importance.csv", index=False)
    print("Lasso feature importance saved\n")

    return to_drop

