"""Checks for the mock-viva examiner's round-2 follow-up questions (real Nagpur weather).

  F-06  cheap sensors (rain gauge + max/min thermometer) with extraterrestrial radiation Ra added
  F-09  what the served Random Forest returns at Kc 1.3 (outside maize's training range) against Kc 1.2
  F-07  TreeSHAP sign for humidity on the 17 Apr 2024 demo day (needs the `shap` package)
  F-02  effective rain from a 10 mm and a 30 mm daily shower: our formula against CROPWAT's USDA-SCS form
  P1-08 recount of synthetic days at the 8 mm/day ET0 cap

    python src/viva_round2_checks.py      -> report/viva_round2_checks.json
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate_dataset as G  # noqa: E402

SPLIT, T, SEEDS, WINDOW = 1600, "irrigation_demand_mm", [0, 1, 2], 7
torch.set_num_threads(2)


# same MLP/LSTM and training loop as src/viva_followups.py (copied, since that script runs on import)
class MLP(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x[:, -1]).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.lstm = nn.LSTM(f, 48, batch_first=True)
        self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


def deep(cls, X, y, tr_end, te_end, X_te=None, y_te=None, window=WINDOW):
    """Train on windows ending before tr_end, test on windows ending in [tr_end, te_end) (or on X_te, y_te); mean MAE of 3 seeds."""
    f = X.shape[1]
    win = lambda A, ends: np.stack([A[i - window + 1:i + 1] for i in ends]).astype(np.float32)  # noqa: E731
    tr = np.arange(WINDOW - 1, tr_end)
    wtr = win(X, tr)
    if X_te is None:
        te = np.arange(tr_end, te_end)
        wte, yte = win(X, te), y[te]
    else:
        te = np.arange(window - 1, len(X_te))
        wte, yte = win(X_te, te), y_te[te]
    mu, sd = wtr.reshape(-1, f).mean(0), wtr.reshape(-1, f).std(0) + 1e-6
    ymu, ysd = y[tr].mean(), y[tr].std()
    res = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xt, yt = torch.tensor((wtr - mu) / sd), torch.tensor(((y[tr] - ymu) / ysd).astype(np.float32))
        nval = int(len(Xt) * 0.1)
        Xf, yf, Xv, yv = Xt[:-nval], yt[:-nval], Xt[-nval:], yt[-nval:]
        m = cls(f)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-4)
        best, state, wait = np.inf, None, 0
        for _ in range(400):
            m.train()
            perm = torch.randperm(len(Xf))
            for b in range(0, len(Xf), 64):
                j = perm[b:b + 64]
                opt.zero_grad()
                nn.functional.mse_loss(m(Xf[j]), yf[j]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(m(Xv), yv).item()
            if v < best - 1e-5:
                best, state, wait = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= 30:
                    break
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            res.append(mae(yte, np.clip(m(torch.tensor((wte - mu) / sd)).numpy() * ysd + ymu, 0, None)))
    return float(np.mean(res)), float(np.std(res))


F7 = ["rainfall_mm", "temperature_C", "humidity_percent", "wind_speed_mps", "solar_radiation_MJ_m2_day", "crop_coefficient", "growth_stage"]
mae = lambda y, p: float(mean_absolute_error(y, p))  # noqa: E731
nasa = pd.read_csv(ROOT / "data/irrigation_dataset_nasa_power.csv", parse_dates=["date"])
raw = pd.read_csv(ROOT / "data/nasa_power_nagpur_raw.csv", parse_dates=["date"])
nasa = nasa.merge(raw[["date", "T2M_MAX", "T2M_MIN"]].rename(columns={"T2M_MAX": "tmax_C", "T2M_MIN": "tmin_C"}), on="date")
nasa["ra_MJ"] = G.extraterrestrial_radiation(nasa.date.dt.day_of_year.values, G.LATITUDE_DEG)
y = nasa[T].values.astype(np.float32)
out = {}

# F-06
cheap = ["rainfall_mm", "tmax_C", "tmin_C", "crop_coefficient", "growth_stage"]
res = {}
for label, feats in [("cheap set", cheap), ("cheap set + Ra", cheap + ["ra_MJ"])]:
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=2).fit(nasa[feats][:SPLIT], y[:SPLIT])
    res[label] = dict(RF=mae(y[SPLIT:], np.clip(rf.predict(nasa[feats][SPLIT:]), 0, None)),
                      LSTM=deep(LSTM, nasa[feats].values.astype(np.float32), y, SPLIT, len(nasa))[0])
    print("F-06", label, res[label], flush=True)
out["cheap_sensors_with_ra"] = res

# F-09
app = joblib.load(ROOT / "models/random_forest_nasa.joblib")
day = dict(rainfall_mm=0, temperature_C=33.7, humidity_percent=43, wind_speed_mps=1.5, solar_radiation_MJ_m2_day=26.7, growth_stage=3)
out["rf_at_kc"] = {str(k): float(app.predict(pd.DataFrame([{**day, "crop_coefficient": k}])[F7])[0]) for k in [1.0, 1.1, 1.2, 1.3, 1.4]}
out["rf_at_kc"]["training_kc_max"] = float(nasa.crop_coefficient[:SPLIT].max())
print("F-09", out["rf_at_kc"], flush=True)

# F-07
try:
    import shap
    x = pd.DataFrame([{**day, "crop_coefficient": 1.2}])[F7]
    sv = shap.TreeExplainer(app).shap_values(x)[0]
    out["treeshap_demo_day"] = {f: float(v) for f, v in zip(F7, sv)}
    print("F-07", out["treeshap_demo_day"], flush=True)
except ImportError:
    out["treeshap_demo_day"] = "shap not installed"

# F-02
ours = lambda P: max(0.0, min(P, 0.8 * P - 25 if P > 75 else 0.6 * P - 10))  # noqa: E731
scs_cropwat = lambda P: P * (125 - 0.2 * P) / 125 if P <= 250 else 125 + 0.1 * P  # noqa: E731
aglw = lambda P: max(0.0, 0.6 * P - 10 if P <= 70 else 0.8 * P - 24)  # noqa: E731
out["effective_rain_mm"] = {str(P): dict(ours=ours(P), usda_scs_cropwat=scs_cropwat(P), fao_aglw=aglw(P)) for P in [5, 10, 20, 30, 80]}
print("F-02", out["effective_rain_mm"], flush=True)

# P1-08 recount
syn = pd.read_csv(ROOT / "data/irrigation_dataset.csv", parse_dates=["date"])
out["synthetic_cap_recount"] = dict(train=int((syn.et0_mm[:SPLIT] >= 7.999).sum()), test=int((syn.et0_mm[SPLIT:] >= 7.999).sum()),
                                    floor_days=int((syn.et0_mm <= 1.001).sum()), real_floor_or_cap="real target: ET0 only clipped at 0")
print("P1-08", out["synthetic_cap_recount"], flush=True)

(ROOT / "report/viva_round2_checks.json").write_text(json.dumps(out, indent=2))
