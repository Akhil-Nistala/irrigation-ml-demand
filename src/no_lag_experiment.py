"""Classical ML vs deep learning on the 7 weather + crop inputs only (yesterday's irrigation demand is never used).

Same chronological 80/20 split and same classical models as model_training.ipynb. Deep-learning models:
  - MLP          : feed-forward network on today's 7 features
  - 1D-CNN, LSTM : sequence models on the last 7 days of the same 7 features (past weather, never past demand)
  - RF + 7-day window : the same 7-day history flattened for a Random Forest, so the sequence models are
                        compared against a classical model that sees identical information

Deep models are trained with 3 seeds (mean +/- std reported), early stopping on the last 10% of the
training period (still chronological), inputs standardised on the training rows only.

    python src/no_lag_experiment.py                                                   # synthetic weather
    python src/no_lag_experiment.py --data data/irrigation_dataset_nasa_power.csv --tag nasa   # real NASA POWER weather

Also records train / validation / test error of every deep model (overfitting check).
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

ROOT = pathlib.Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
SEEDS = [0, 1, 2]
WINDOW = 7
torch.set_num_threads(2)

FEATURES = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps",
            "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
TARGET = "irrigation_demand_mm"
LABELS = {"rainfall_mm": "Rainfall", "temperature_C": "Temperature", "humidity_percent": "Humidity",
          "wind_speed_mps": "Wind speed", "solar_radiation_MJ_m2_day": "Solar radiation",
          "crop_coefficient": "Crop coefficient", "growth_stage": "Growth stage"}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/irrigation_dataset.csv")
ap.add_argument("--tag", default="no_lag")
args = ap.parse_args()
TAG = args.tag
DATA_LABEL = {"no_lag": "synthetic weather", "nasa": "NASA POWER weather"}.get(TAG, TAG)
FIGS = {"no_lag": ["09_classical_models", "10_ml_vs_dl_no_lag", "11_actual_vs_predicted_no_lag", "12_feature_importance_no_lag",
                   "09b_lr_vs_rf_actual_vs_predicted"],
        "nasa": ["14_nasa_classical_models", "15_nasa_ml_vs_dl", "16_nasa_actual_vs_predicted", "17_nasa_feature_importance",
                 "14b_nasa_lr_vs_rf_actual_vs_predicted"]
        }.get(TAG) or [f"{TAG}_{i}" for i in ("classical_models", "ml_vs_dl", "actual_vs_predicted", "feature_importance", "lr_vs_rf")]
FIGS = [ROOT / "figures" / f"{f}.png" for f in FIGS]

df = pd.read_csv(ROOT / args.data, parse_dates=["date"])
split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]
X_train, y_train = train_df[FEATURES], train_df[TARGET].values
X_test, y_test = test_df[FEATURES], test_df[TARGET].values


def scores(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p),
                negative_predictions=int((p < 0).sum()))


# ---------------- classical models (identical hyper-parameters to the original notebook) ----------------
classical = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=2),
}
results, preds = {}, {}
for name, m in classical.items():
    m.fit(X_train, y_train)
    preds[name] = m.predict(X_test)
    results[name] = dict(family="Classical ML", **scores(y_test, preds[name]))

# ---------------- windowed inputs: day t uses features of days t-6..t ----------------
Xall = df[FEATURES].values.astype(np.float32)
idx = np.arange(WINDOW - 1, len(df))                       # first 6 days have no full window
win = np.stack([Xall[i - WINDOW + 1:i + 1] for i in idx])  # (n, 7, 7)
yall = df[TARGET].values.astype(np.float32)[idx]
tr_mask, te_mask = idx < split, idx >= split               # test days are exactly the 400 test days above
assert te_mask.sum() == len(test_df)

rf_win = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=2)
rf_win.fit(win[tr_mask].reshape(tr_mask.sum(), -1), yall[tr_mask])
preds["RF + 7-day window"] = rf_win.predict(win[te_mask].reshape(te_mask.sum(), -1))
results["RF + 7-day window"] = dict(family="Classical ML", **scores(y_test, preds["RF + 7-day window"]))

# ---------------- deep learning ----------------
mu, sd = Xall[:split].mean(0), Xall[:split].std(0) + 1e-6
ymu, ysd = yall[tr_mask].mean(), yall[tr_mask].std()
win_s = (win - mu) / sd


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):                 # x: (b, 7, f) -> uses today only
        return self.net(x[:, -1]).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(len(FEATURES), 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 * WINDOW, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)).flatten(1)).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(len(FEATURES), 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def train_deep(cls, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = torch.tensor(win_s[tr_mask]), torch.tensor((yall[tr_mask] - ymu) / ysd)
    nval = int(len(Xtr) * 0.1)                              # last 10% of the training period = validation
    Xfit, yfit, Xval, yval = Xtr[:-nval], ytr[:-nval], Xtr[-nval:], ytr[-nval:]
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(400):
        model.train()
        perm = torch.randperm(len(Xfit))
        for b in range(0, len(Xfit), 64):
            j = perm[b:b + 64]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xfit[j]), yfit[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xval), yval).item()
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 30:
                break
    model.load_state_dict(best_state)
    model.eval()
    unscale = lambda X: model(X).numpy() * ysd + ymu  # noqa: E731
    with torch.no_grad():
        p = unscale(torch.tensor(win_s[te_mask]))
        fit = dict(train=scores(yfit.numpy() * ysd + ymu, unscale(Xfit)), validation=scores(yval.numpy() * ysd + ymu, unscale(Xval)),
                   test=scores(yall[te_mask], p))
    return p, epoch + 1, sum(t.numel() for t in model.parameters()), fit


overfit_rows = []
for name, cls in [("MLP", MLP), ("1D-CNN (7-day)", CNN1D), ("LSTM (7-day)", LSTM)]:
    runs = [train_deep(cls, s) for s in SEEDS]
    per_seed = [scores(y_test, r[0]) for r in runs]
    preds[name] = np.mean([r[0] for r in runs], axis=0)     # seed-averaged prediction for the plots
    results[name] = dict(family="Deep learning",
                         **{k: float(np.mean([s[k] for s in per_seed])) for k in ("MAE", "RMSE", "R2")},
                         **{k + "_std": float(np.std([s[k] for s in per_seed])) for k in ("MAE", "RMSE", "R2")},
                         negative_predictions=int(np.mean([s["negative_predictions"] for s in per_seed])),
                         epochs=[r[1] for r in runs], parameters=runs[0][2])
    for seed, r in zip(SEEDS, runs):
        overfit_rows.append(dict(model=name, seed=seed, epochs=r[1], **{f"{part}_{m}": r[3][part][m] for part in ("train", "validation", "test")
                                                                         for m in ("MAE", "R2")}))
    print(name, {k: round(v, 3) for k, v in results[name].items() if isinstance(v, float)}, flush=True)
pd.DataFrame(overfit_rows).to_csv(ROOT / f"report/{TAG}_overfitting_check.csv", index=False)

res = pd.DataFrame(results).T
res.index.name = "Model"
for k in ("MAE", "RMSE", "R2"):
    res[k] = res[k].astype(float)
res.to_csv(ROOT / f"report/{TAG}_model_comparison.csv")
(ROOT / f"models/metrics_{TAG}.json").write_text(json.dumps(
    dict(data=args.data, features=FEATURES, results=results),
    indent=2, default=float))

# ---------------- permutation importance for the 7-feature Random Forest ----------------
rf = classical["Random Forest"]
pi = permutation_importance(rf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="r2", n_jobs=2)
imp = pd.DataFrame({"correlation_with_target": df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET),
                    "rf_mdi_importance": pd.Series(rf.feature_importances_, index=FEATURES),
                    "rf_permutation_importance_mean": pd.Series(pi.importances_mean, index=FEATURES),
                    "rf_permutation_importance_std": pd.Series(pi.importances_std, index=FEATURES)}
                   ).sort_values("rf_permutation_importance_mean", ascending=False)
imp.to_csv(ROOT / f"report/{TAG}_feature_importance.csv")

# ---------------- figures ----------------
TEAL, AMBER, LEAF, GREY = "#0B4F6C", "#E8A33D", "#3B8C6E", "#9AA5A8"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# classical models only: MAE / RMSE / R2 bars
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
names = list(classical)
for ax, metric, title in zip(axes, ["MAE", "RMSE", "R2"], ["MAE (mm/day), lower is better", "RMSE (mm/day), lower is better",
                                                           "R$^2$, higher is better"]):
    v = [results[n][metric] for n in names]
    ax.bar(names, v, color=[GREY, LEAF, TEAL], width=0.6)
    for i, val in enumerate(v):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=9)
    if metric == "R2":
        ax.set_ylim(min(0.8, min(v) - 0.03), 1.0)
plt.tight_layout()
plt.savefig(FIGS[0], dpi=150)
plt.close()

# Linear Regression vs Random Forest, actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, n, c in zip(axes, ["Linear Regression", "Random Forest"], [GREY, TEAL]):
    ax.scatter(y_test, preds[n], s=14, alpha=0.5, color=c)
    lo = min(0, preds[n].min()) - 0.5
    lim = [lo, max(y_test.max(), preds[n].max()) + 0.5]
    ax.plot(lim, lim, "--", color="#C0392B", lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Actual demand (mm/day)")
    ax.set_ylabel("Predicted demand (mm/day)")
    ax.set_title(f"{n}: MAE {results[n]['MAE']:.2f}, R$^2$ {results[n]['R2']:.3f}, {results[n]['negative_predictions']} negative")
plt.tight_layout()
plt.savefig(FIGS[4], dpi=150)
plt.close()

# all models
order = res.sort_values("MAE", ascending=False).index
fig, ax = plt.subplots(figsize=(10, 4.8))
colors = [AMBER if res.loc[n, "family"] == "Deep learning" else TEAL for n in order]
err = [res.loc[n].get("MAE_std", 0) if res.loc[n, "family"] == "Deep learning" else 0 for n in order]
err = [0 if pd.isna(e) else e for e in err]
ax.barh(order, res.loc[order, "MAE"], color=colors)
dl = [i for i, e in enumerate(err) if e]
ax.errorbar(res.loc[order, "MAE"].values[dl], dl, xerr=np.array(err)[dl], fmt="none", ecolor="black", capsize=3)
for i, n in enumerate(order):
    ax.text(res.loc[n, "MAE"] + err[i] + 0.015, i, f"MAE {res.loc[n, 'MAE']:.3f}  |  R$^2$ {res.loc[n, 'R2']:.3f}", va="center", fontsize=9)
ax.set_xlabel("Test MAE (mm/day), lower is better")
ax.set_xlim(0, res["MAE"].max() * 1.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=AMBER)],
          labels=["Classical ML", "Deep learning (mean of 3 seeds, $\\pm$1 sd)"], frameon=False, loc="upper right")
ax.set_title(f"All models, 7 weather + crop inputs, {DATA_LABEL}")
plt.tight_layout()
plt.savefig(FIGS[1], dpi=150)
plt.close()

# actual vs predicted, best classical vs best deep model
best_dl = res[res.family == "Deep learning"]["MAE"].idxmin()
best_cl = res[res.family == "Classical ML"]["MAE"].idxmin()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, n, c in zip(axes, [best_cl, best_dl], [TEAL, AMBER]):
    ax.scatter(y_test, preds[n], s=14, alpha=0.5, color=c)
    lim = [0, max(y_test.max(), preds[n].max()) + 0.5]
    ax.plot(lim, lim, "--", color="#C0392B", lw=1.4)
    ax.set_xlabel("Actual demand (mm/day)")
    ax.set_ylabel("Predicted demand (mm/day)")
    ax.set_title(f"{n}: MAE {results[n]['MAE']:.2f}, R$^2$ {results[n]['R2']:.3f}")
plt.tight_layout()
plt.savefig(FIGS[2], dpi=150)
plt.close()

# feature importance
o = imp.sort_values("rf_permutation_importance_mean").index
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].barh([LABELS[f] for f in o], imp.loc[o, "rf_mdi_importance"], color=LEAF)
axes[0].set_title("Random Forest importance (training splits)")
axes[1].barh([LABELS[f] for f in o], imp.loc[o, "rf_permutation_importance_mean"], xerr=imp.loc[o, "rf_permutation_importance_std"],
             color=TEAL, capsize=3)
axes[1].set_title("Permutation importance (drop in test R$^2$)")
plt.tight_layout()
plt.savefig(FIGS[3], dpi=150)
plt.close()

print(res[["family", "MAE", "RMSE", "R2", "negative_predictions"]].round(3).to_string())
print(imp.round(3).to_string())
