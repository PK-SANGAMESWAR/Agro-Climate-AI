# =============================================================================
# 24AI636 DL — Mini-Project 2  |  ALL-IN-ONE NOTEBOOK
# Pretrained CNN + Temporal Modeling using RNN / LSTM / GRU / Transformer
# =============================================================================
# ✅ Feature Extraction using ≥2 Pretrained CNNs
# ✅ Fine-tuning Pretrained CNN
# ✅ Temporal Data Preprocessing Pipeline
# ✅ RNN / LSTM / GRU implementation
# ✅ Attention-based model
# ✅ Embedding usage
# ✅ Hyperparameter experimentation
# ✅ Model comparison & evaluation metrics
# ✅ Clean, modular, documented code
# =============================================================================

# ─── CELL 1 : IMPORTS ────────────────────────────────────────────────────────
import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ─── CELL 2 : PATH RESOLVER ──────────────────────────────────────────────────
def resolve_data_dir() -> str:
    """
    Auto-detect DATA_DIR for both Jupyter and .py scripts.

    Expected layout:
        Agro-Climate/
        └── data/
            └── 10_4231_R72F7KK2/
                └── bundle/
                    └── Agro-Climatic Data by County/   ← DATA_DIR
                        ├── gddAprOct.csv
                        ├── pptAprOct.csv
                        ├── yielddata.csv
                        ├── gridInfo.csv
                        ├── soil2011.csv
                        └── cntymap/
                                ├── cntymap.shp
                                ├── cntymap.dbf
                                └── cntymap.shx
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base = os.getcwd()          # Jupyter: use working directory

    candidates = [
        os.path.join(base, 'data', '10_4231_R72F7KK2', 'bundle',
                     'Agro-Climatic Data by County'),
        os.path.join(base, 'bundle', 'Agro-Climatic Data by County'),
        os.path.join(base, 'Agro-Climatic Data by County'),
        os.path.join('data', '10_4231_R72F7KK2', 'bundle',
                     'Agro-Climatic Data by County'),
    ]
    for p in candidates:
        if os.path.isdir(p):
            print(f"✅ DATA_DIR found:\n   {os.path.abspath(p)}\n")
            return p

    print("❌ Auto-detect failed. Searched:")
    for p in candidates:
        print(f"   {p}")
    print("\n👉 Set DATA_DIR manually in the next cell.")
    return candidates[0]


DATA_DIR = resolve_data_dir()
# ── If auto-detect failed, uncomment and edit this line: ──
# DATA_DIR = r'C:\Users\LOQ\Downloads\Agro-Climate\data\10_4231_R72F7KK2\bundle\Agro-Climatic Data by County'


# ─── CELL 3 : DATA LOADING & MERGING ─────────────────────────────────────────
# WHY MERGE?
#   yielddata  → target variable          (county_id + year)
#   gddAprOct  → monthly GDD sequence     (county_id + year)
#   pptAprOct  → monthly PPT sequence     (county_id + year)
#   soil2011   → static soil properties   (county_id only)
#   gridInfo   → lat/lon spatial coords   (county_id only)

def _detect_cols(df, name):
    """Auto-detect county_id / year / yield column names."""
    cl = {c.lower().replace('_','').replace('-',''): c for c in df.columns}
    id_col    = next((cl[a] for a in ['countyid','fips','geoid','county','id']
                      if a in cl), df.columns[0])
    year_col  = next((cl[a] for a in ['year','yr','cropyear'] if a in cl), None)
    yield_col = next((cl[a] for a in ['yield','corn','cornyield','yldbu',
                                       'value','grainyield'] if a in cl), None)
    if year_col is None:
        for c in df.columns:
            try:
                v = df[c].dropna().astype(int)
                if v.between(1970, 2030).all() and c != id_col:
                    year_col = c; break
            except: pass
    print(f"  [{name}] id={id_col!r}  year={year_col!r}  yield={yield_col!r}")
    return id_col, year_col, yield_col


def _std(df, id_col, year_col=None, yield_col=None):
    """Rename detected columns to standard names."""
    m = {}
    if id_col    and id_col    != 'county_id': m[id_col]    = 'county_id'
    if year_col  and year_col  != 'year':      m[year_col]  = 'year'
    if yield_col and yield_col != 'yield':     m[yield_col] = 'yield'
    return df.rename(columns=m)


def load_and_merge(data_dir):
    """Load all CSVs, auto-rename columns, merge into one master DataFrame."""
    print("="*60)
    print("STEP 1 — DATA LOADING & MERGING")
    print("="*60)

    yield_df = pd.read_csv(os.path.join(data_dir, 'yielddata.csv'))
    gdd_df   = pd.read_csv(os.path.join(data_dir, 'gddAprOct.csv'))
    ppt_df   = pd.read_csv(os.path.join(data_dir, 'pptAprOct.csv'))
    soil_df  = pd.read_csv(os.path.join(data_dir, 'soil2011.csv'))
    grid_df  = pd.read_csv(os.path.join(data_dir, 'gridInfo.csv'))

    print("\n--- Auto-detecting column names ---")
    yield_df = _std(yield_df, *_detect_cols(yield_df, 'yielddata'))
    gdd_df   = _std(gdd_df,   *_detect_cols(gdd_df,   'gddAprOct'))
    ppt_df   = _std(ppt_df,   *_detect_cols(ppt_df,   'pptAprOct'))
    soil_df  = _std(soil_df,  *_detect_cols(soil_df,  'soil2011'))
    grid_df  = _std(grid_df,  *_detect_cols(grid_df,  'gridInfo'))

    for tag, dframe, reqs in [
        ('yielddata', yield_df, ['county_id','year','yield']),
        ('gddAprOct', gdd_df,   ['county_id','year']),
        ('pptAprOct', ppt_df,   ['county_id','year']),
        ('soil2011',  soil_df,  ['county_id']),
        ('gridInfo',  grid_df,  ['county_id']),
    ]:
        miss = [c for c in reqs if c not in dframe.columns]
        if miss:
            raise ValueError(f"[{tag}] Still missing {miss}.\n"
                             f"Actual cols: {list(dframe.columns)}")

    # Prefix monthly columns
    gdd_m = [c for c in gdd_df.columns if c not in ('county_id','year')]
    ppt_m = [c for c in ppt_df.columns if c not in ('county_id','year')]
    soil_m= [c for c in soil_df.columns if c != 'county_id']
    gdd_df.rename(columns={c: f'gdd_{c}'  for c in gdd_m},  inplace=True)
    ppt_df.rename(columns={c: f'ppt_{c}'  for c in ppt_m},  inplace=True)
    soil_df.rename(columns={c: f'soil_{c}' for c in soil_m}, inplace=True)

    print("\n--- Merging ---")
    df = yield_df.merge(gdd_df,  on=['county_id','year'], how='inner')
    print(f"  yield + GDD  : {df.shape}")
    df = df.merge(ppt_df,  on=['county_id','year'], how='inner')
    print(f"  + PPT        : {df.shape}")
    df = df.merge(soil_df, on='county_id',           how='left')
    print(f"  + Soil       : {df.shape}")
    df = df.merge(grid_df, on='county_id',           how='left')
    print(f"  + Grid       : {df.shape}")

    df.dropna(subset=['yield'], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"\n✅ Master DataFrame: {df.shape}")
    print(f"   Columns: {list(df.columns)}\n")
    return df


DF = load_and_merge(DATA_DIR)


# ─── CELL 4 : TEMPORAL PREPROCESSING PIPELINE ────────────────────────────────
class TemporalPreprocessor:
    """
    Builds (N, T, F) sequences from the merged DataFrame.
      X_seq     : (N, months, 2)   monthly [GDD, PPT]
      X_static  : (N, S)           soil + lat/lon
      county_ids: (N,)             integer index for embedding layer
      y         : (N,)             corn yield target
    """
    def __init__(self):
        self.seq_scaler    = StandardScaler()
        self.static_scaler = StandardScaler()
        self.y_scaler      = MinMaxScaler()
        self.n_counties    = None
        self.month_labels  = []

    def build(self, df):
        """
        Real column structure discovered from the actual CSVs:
        ───────────────────────────────────────────────────────
        gddAprOct.csv:
          gdd0         → base-year GDD (scalar)
          gddm1…gddm60 → GDD for past 60 months  ← TEMPORAL SEQUENCE
          gddp1…gddp60 → GDD for future 60 months (treated as static)

        pptAprOct.csv:
          ppt          → single total precipitation value (static scalar)

        Strategy:
          • X_seq    ← gddm columns sorted by lag (gddm60 oldest → gddm1 newest)
          • X_static ← gdd0 + gddp cols + ppt + soil + lat/lon
        """
        import re

        all_cols = list(df.columns)

        # ── Temporal sequence: gddm1…gddmN (past months, sorted oldest→newest) ──
        gddm_cols = [c for c in all_cols if re.match(r'^gdd_gddm\d+$', c)]
        gddm_cols = sorted(gddm_cols,
                           key=lambda c: int(re.search(r'\d+', c).group()),
                           reverse=True)   # gddm60 first (oldest), gddm1 last

        # ── Static GDD features: gdd0 + gddpX (future projections) ──────────
        gdd0_cols  = [c for c in all_cols if re.match(r'^gdd_gdd0$', c)]
        gddp_cols  = [c for c in all_cols if re.match(r'^gdd_gddp\d+$', c)]

        # ── PPT (single value or multiple) ────────────────────────────────────
        ppt_cols   = [c for c in all_cols if c.startswith('ppt_')]

        # ── Soil and geo ──────────────────────────────────────────────────────
        soil_cols  = [c for c in all_cols if c.startswith('soil_')]
        geo_cols   = [c for c in ['lat','lon'] if c in all_cols]

        print(f"  Temporal sequence (gddm past months) : {len(gddm_cols)} steps")
        print(f"  Static GDD features (gdd0 + gddp)    : {len(gdd0_cols)+len(gddp_cols)}")
        print(f"  PPT features                          : {len(ppt_cols)}")
        print(f"  Soil features                         : {len(soil_cols)}")
        print(f"  Geo features                          : {len(geo_cols)}")

        if len(gddm_cols) == 0:
            # Fallback: if naming differs, use ALL gdd_ columns as sequence
            gddm_cols = sorted(c for c in all_cols if c.startswith('gdd_'))
            print(f"  ⚠️  No gddm* cols found — using all {len(gddm_cols)} gdd_ cols as sequence")

        # ── Build X_seq : (N, T, 1)  — one feature per timestep ─────────────
        gddm_vals = df[gddm_cols].values.astype(np.float32)   # (N, T)
        X_seq     = gddm_vals[:, :, np.newaxis]                # (N, T, 1)

        # ── Build X_static : gdd0 + gddp + ppt + soil + geo ─────────────────
        static_cols = gdd0_cols + gddp_cols + ppt_cols + soil_cols + geo_cols
        X_static = (df[static_cols].values.astype(np.float32)
                    if static_cols else np.zeros((len(df),1), np.float32))

        # ── County embedding index ────────────────────────────────────────────
        cats = df['county_id'].astype('category')
        county_ids = cats.cat.codes.values.astype(np.int64)
        self.n_counties   = int(county_ids.max()) + 1
        self.month_labels = [c.replace('gdd_gddm','m') for c in gddm_cols]

        y = df['yield'].values.astype(np.float32)
        print(f"  X_seq    : {X_seq.shape}  (samples, past_months, 1)")
        print(f"  ✅ X_static : {X_static.shape}  (gdd0+gddp+ppt+soil+geo)")
        return X_seq, X_static, county_ids, y

    def fit_transform(self, X_seq, X_static, y):
        N, T, F = X_seq.shape
        Xs  = self.seq_scaler.fit_transform(
                  X_seq.reshape(-1,F)).reshape(N,T,F)
        Xst = self.static_scaler.fit_transform(X_static)
        ys  = self.y_scaler.fit_transform(y.reshape(-1,1)).ravel()
        return Xs, Xst, ys

    def transform(self, X_seq, X_static, y):
        N, T, F = X_seq.shape
        Xs  = self.seq_scaler.transform(X_seq.reshape(-1,F)).reshape(N,T,F)
        Xst = self.static_scaler.transform(X_static)
        ys  = self.y_scaler.transform(y.reshape(-1,1)).ravel()
        return Xs, Xst, ys

    def inverse_y(self, y):
        return self.y_scaler.inverse_transform(
                   np.array(y).reshape(-1,1)).ravel()


def temporal_split(df, X_seq, X_static, county_ids, y, val_frac=0.2):
    """Split by year (no leakage): last val_frac of years → validation."""
    years  = sorted(df['year'].unique())
    cutoff = years[int(len(years)*(1-val_frac))]
    mask   = df['year'].values < cutoff
    return (X_seq[mask],    X_seq[~mask],
            X_static[mask], X_static[~mask],
            county_ids[mask], county_ids[~mask],
            y[mask], y[~mask])


class CornDataset(Dataset):
    def __init__(self, X_seq, X_static, county_ids, y):
        self.X_seq      = torch.tensor(X_seq,       dtype=torch.float32)
        self.X_static   = torch.tensor(X_static,    dtype=torch.float32)
        self.county_ids = torch.tensor(county_ids,  dtype=torch.long)
        self.y          = torch.tensor(y,            dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):
        return self.X_seq[i], self.X_static[i], self.county_ids[i], self.y[i]


def compute_metrics(y_true, y_pred, label=''):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    if label:
        print(f"\n  {'─'*38}")
        print(f"  Model  : {label}")
        print(f"  RMSE   : {rmse:.4f}")
        print(f"  MAE    : {mae:.4f}")
        print(f"  R²     : {r2:.4f}")
        print(f"  MAPE   : {mape:.2f}%")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


# Build preprocessed data (shared by all models)
PREP   = TemporalPreprocessor()
X_seq, X_static, county_ids, y = PREP.build(DF)

(Xs_tr, Xs_vl, Xst_tr, Xst_vl,
 cid_tr, cid_vl, y_tr, y_vl) = temporal_split(DF, X_seq, X_static, county_ids, y)

Xs_tr, Xst_tr, y_tr = PREP.fit_transform(Xs_tr, Xst_tr, y_tr)
Xs_vl, Xst_vl, y_vl = PREP.transform(Xs_vl, Xst_vl, y_vl)

TR_DS = CornDataset(Xs_tr, Xst_tr, cid_tr, y_tr)
VL_DS = CornDataset(Xs_vl, Xst_vl, cid_vl, y_vl)

SEQ_FEAT    = Xs_tr.shape[2]
STATIC_FEAT = Xst_tr.shape[1]
N_COUNTIES  = PREP.n_counties

print(f"Train samples : {len(TR_DS)}")
print(f"Val   samples : {len(VL_DS)}")
print(f"Sequence shape: (N, {Xs_tr.shape[1]}, {SEQ_FEAT})")
print(f"Static shape  : {STATIC_FEAT}")
print(f"N counties    : {N_COUNTIES}")
print(f"Month labels  : {PREP.month_labels}")


# ─── CELL 5 : SHARED TRAINING LOOP ───────────────────────────────────────────
def run_training(model, tr_ds, vl_ds, prep,
                 batch_size=256, epochs=60, lr=1e-3,
                 patience=10, label='model',
                 optimizer_cls=optim.Adam, scheduler='plateau'):
    """Generic training loop used by LSTM, CNN and Transformer models."""
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    vl_ld = DataLoader(vl_ds, batch_size=batch_size)
    crit  = nn.MSELoss()
    opt   = optimizer_cls(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=lr, weight_decay=1e-4)
    if scheduler == 'plateau':
        sched = optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=0.5, patience=5)
    else:
        sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=epochs, eta_min=1e-5)

    hist = {'train_loss':[], 'val_loss':[], 'val_rmse':[], 'val_r2':[]}
    best_val, wait = float('inf'), 0

    for ep in range(1, epochs+1):
        # ── train ──
        model.train(); tl = 0
        for batch in tr_ld:
            inputs = [t.to(DEVICE) for t in batch[:-1]]
            ytrue  = batch[-1].to(DEVICE)
            opt.zero_grad()
            pred   = model(*inputs)
            if isinstance(pred, tuple): pred = pred[0]   # models returning (pred, attn)
            loss   = crit(pred, ytrue)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
        tl /= len(tr_ld)

        # ── validate ──
        model.eval(); vl = 0
        preds, truths = [], []
        with torch.no_grad():
            for batch in vl_ld:
                inputs = [t.to(DEVICE) for t in batch[:-1]]
                ytrue  = batch[-1]
                out    = model(*inputs)
                if isinstance(out, tuple): out = out[0]
                vl += crit(out, ytrue.to(DEVICE)).item()
                preds.extend(out.cpu().numpy())
                truths.extend(ytrue.numpy())
        vl /= len(vl_ld)
        p  = prep.inverse_y(np.array(preds))
        t  = prep.inverse_y(np.array(truths))
        rmse = np.sqrt(mean_squared_error(t, p))
        r2   = r2_score(t, p)

        hist['train_loss'].append(tl)
        hist['val_loss'].append(vl)
        hist['val_rmse'].append(rmse)
        hist['val_r2'].append(r2)

        if scheduler == 'plateau': sched.step(vl)
        else:                      sched.step()

        if ep % 10 == 0:
            print(f"  [{label}] Ep {ep:3d} | "
                  f"TrLoss={tl:.4f}  VaLoss={vl:.4f}  "
                  f"RMSE={rmse:.3f}  R²={r2:.4f}")

        if vl < best_val:
            best_val, wait = vl, 0
            torch.save(model.state_dict(), f'/tmp/best_{label}.pt')
        else:
            wait += 1
            if wait >= patience:
                print(f"  ⏹ Early stop at epoch {ep}")
                break

    model.load_state_dict(torch.load(f'/tmp/best_{label}.pt'))
    return hist, model


def final_eval(model, vl_ds, prep, batch_size=256, label=''):
    """Evaluate a trained model and return metrics dict."""
    vl_ld = DataLoader(vl_ds, batch_size=batch_size)
    preds, truths = [], []
    model.eval()
    with torch.no_grad():
        for batch in vl_ld:
            inputs = [t.to(DEVICE) for t in batch[:-1]]
            out = model(*inputs)
            if isinstance(out, tuple): out = out[0]
            preds.extend(out.cpu().numpy())
            truths.extend(batch[-1].numpy())
    p = prep.inverse_y(np.array(preds))
    t = prep.inverse_y(np.array(truths))
    return compute_metrics(t, p, label=label)


# ─── CELL 6 : MODEL 1 — RNN / LSTM / GRU  +  ATTENTION  +  EMBEDDING ────────
# ===========================================================================

class BahdanauAttention(nn.Module):
    """
    Additive attention over all LSTM hidden states.
    Learns WHICH months matter most (e.g. July heat stress).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1,           bias=False)

    def forward(self, h):                          # h: (B, T, H)
        score   = torch.tanh(self.W(h))            # (B, T, H)
        weights = torch.softmax(self.v(score), dim=1)  # (B, T, 1)
        context = (weights * h).sum(dim=1)         # (B, H)
        return context, weights.squeeze(-1)        # (B,H), (B,T)


class TemporalModel(nn.Module):
    """
    RNN / LSTM / GRU with:
      • Bahdanau attention        (which months drive yield?)
      • County embedding          (learnable spatial identity)
      • Static soil/geo branch    (fused into prediction head)
    """
    def __init__(self, seq_feat, static_feat, n_counties,
                 hidden_dim=64, n_layers=2,
                 rnn_type='LSTM', embed_dim=16, dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type

        # Embedding: each county → dense vector  (EMBEDDING USAGE)
        self.county_embed = nn.Embedding(n_counties, embed_dim)

        # RNN core
        cfg = dict(input_size=seq_feat, hidden_size=hidden_dim,
                   num_layers=n_layers, batch_first=True,
                   dropout=dropout if n_layers > 1 else 0.0)
        self.rnn = {'LSTM': nn.LSTM, 'GRU': nn.GRU,
                    'RNN':  nn.RNN}[rnn_type](**cfg)

        # Attention  (ATTENTION-BASED MODEL)
        self.attention = BahdanauAttention(hidden_dim)

        # Static branch
        self.static_fc = nn.Sequential(
            nn.Linear(static_feat, 32), nn.ReLU(), nn.Dropout(dropout))

        # Fusion + prediction head
        fuse = hidden_dim + embed_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fuse, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,  64),  nn.ReLU(),
            nn.Linear(64,    1))

    def forward(self, x_seq, x_static, county_ids):
        rnn_out, _  = self.rnn(x_seq)
        context, aw = self.attention(rnn_out)
        embed       = self.county_embed(county_ids)
        static      = self.static_fc(x_static)
        fused       = torch.cat([context, embed, static], dim=1)
        return self.head(fused).squeeze(-1), aw


# ── Hyperparameter search  (HYPERPARAMETER EXPERIMENTATION) ─────────────────
print("\n" + "="*60)
print("MODEL 1 — HYPERPARAMETER SEARCH  (LSTM  lr × hidden_dim)")
print("="*60)

hp_lstm_results = []
for lr in [1e-3, 5e-4]:
    for hd in [64, 128]:
        lbl = f'LSTM_lr{lr}_h{hd}'
        print(f"\n  lr={lr}  hidden={hd}")
        m = TemporalModel(SEQ_FEAT, STATIC_FEAT, N_COUNTIES,
                          hidden_dim=hd, rnn_type='LSTM').to(DEVICE)
        m.rnn_type = lbl
        hist, m = run_training(m, TR_DS, VL_DS, PREP,
                                epochs=30, lr=lr, patience=7, label=lbl)
        hp_lstm_results.append({'lr': lr, 'hidden_dim': hd,
                                 'best_R2':   round(max(hist['val_r2']),   4),
                                 'best_RMSE': round(min(hist['val_rmse']), 4)})

HP_LSTM_DF = pd.DataFrame(hp_lstm_results)
print("\n  LSTM Hyperparameter Results:")
print(HP_LSTM_DF.to_string(index=False))
best_lstm_row = HP_LSTM_DF.loc[HP_LSTM_DF['best_R2'].idxmax()]
BEST_LR_LSTM  = float(best_lstm_row['lr'])
BEST_HD_LSTM  = int(best_lstm_row['hidden_dim'])
print(f"\n  ✅ Best: lr={BEST_LR_LSTM}  hidden={BEST_HD_LSTM}")


# ── Compare RNN vs LSTM vs GRU  (MODEL COMPARISON) ──────────────────────────
print("\n" + "="*60)
print("MODEL 1 — COMPARISON: RNN  vs  LSTM  vs  GRU")
print("="*60)

lstm_metrics   = {}
lstm_histories = {}

for rtype in ['RNN', 'LSTM', 'GRU']:
    print(f"\n  ── {rtype} ──")
    model = TemporalModel(SEQ_FEAT, STATIC_FEAT, N_COUNTIES,
                          hidden_dim=BEST_HD_LSTM, rnn_type=rtype).to(DEVICE)
    hist, model = run_training(model, TR_DS, VL_DS, PREP,
                                epochs=60, lr=BEST_LR_LSTM,
                                patience=10, label=rtype)
    lstm_histories[rtype] = hist
    lstm_metrics[rtype]   = final_eval(model, VL_DS, PREP, label=rtype)

    # Keep LSTM model for attention plot
    if rtype == 'LSTM':
        BEST_LSTM_MODEL = model


# ─── CELL 7 : MODEL 1 PLOTS ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {'RNN':'#e74c3c', 'LSTM':'#2ecc71', 'GRU':'#3498db'}

# Loss curves
for m, h in lstm_histories.items():
    axes[0].plot(h['train_loss'], '--', color=colors[m], alpha=.5)
    axes[0].plot(h['val_loss'],         color=colors[m], label=m)
axes[0].set(title='RNN/LSTM/GRU — Loss', xlabel='Epoch', ylabel='MSE')
axes[0].legend(); axes[0].grid(alpha=.3)

# R² curves
for m, h in lstm_histories.items():
    axes[1].plot(h['val_r2'], color=colors[m], label=m)
axes[1].set(title='RNN/LSTM/GRU — Val R²', xlabel='Epoch', ylabel='R²')
axes[1].legend(); axes[1].grid(alpha=.3)

# Metrics bar chart
met_names = ['RMSE','MAE','R2','MAPE']
x = np.arange(len(met_names)); w = 0.25
for i, (rtype, col) in enumerate(colors.items()):
    vals = [lstm_metrics[rtype][k] for k in met_names]
    axes[2].bar(x + i*w, vals, w, label=rtype, color=col, edgecolor='k')
axes[2].set_xticks(x + w); axes[2].set_xticklabels(met_names)
axes[2].set_title('RNN/LSTM/GRU — Metrics'); axes[2].legend()
axes[2].grid(axis='y', alpha=.3)

plt.suptitle('MODEL 1: RNN / LSTM / GRU', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('lstm_results.png', dpi=150); plt.show()

# Attention weights (LSTM)
vl_ld = DataLoader(VL_DS, batch_size=256)
BEST_LSTM_MODEL.eval()
with torch.no_grad():
    xs, xst, cids, _ = next(iter(vl_ld))
    _, aw = BEST_LSTM_MODEL(xs.to(DEVICE), xst.to(DEVICE), cids.to(DEVICE))
avg_attn = aw.mean(0).cpu().numpy()
plt.figure(figsize=(10, 4))
plt.bar(PREP.month_labels, avg_attn, color='steelblue', edgecolor='k')
plt.title('Attention Weights — Which Months Drive Corn Yield?', fontsize=13)
plt.xlabel('Month'); plt.ylabel('Avg Attention Weight')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig('lstm_attention.png', dpi=150); plt.show()


# ─── CELL 8 : MODEL 2 — PRETRAINED CNN ───────────────────────────────────────
# ===========================================================================

RESIZE = T.Resize((224, 224), antialias=True)

def build_spatial_images(df, feature_col, grid_size=64):
    """Rasterise county-level feature to 2D grid image, one per year."""
    has_geo = 'lat' in df.columns and 'lon' in df.columns
    images  = []
    for yr in sorted(df['year'].unique()):
        ydf  = df[df['year'] == yr]
        grid = np.zeros((grid_size, grid_size), np.float32)
        cnt  = np.zeros_like(grid)
        if has_geo:
            lat0,lat1 = df['lat'].min(), df['lat'].max()
            lon0,lon1 = df['lon'].min(), df['lon'].max()
            for _, row in ydf.iterrows():
                xi = int((row['lon']-lon0)/(lon1-lon0+1e-9)*(grid_size-1))
                yi = int((row['lat']-lat0)/(lat1-lat0+1e-9)*(grid_size-1))
                grid[yi,xi] += row[feature_col]; cnt[yi,xi] += 1
            grid = np.where(cnt>0, grid/cnt, 0)
        else:
            vals = ydf[feature_col].values
            n    = min(len(vals), grid_size*grid_size)
            grid.ravel()[:n] = vals[:n]
        vmin,vmax = grid.min(), grid.max()
        if vmax > vmin: grid = (grid-vmin)/(vmax-vmin)
        images.append(np.stack([grid,grid,grid], axis=0))
    return torch.tensor(np.array(images), dtype=torch.float32)


class CNNExtractor(nn.Module):
    """Pretrained backbone (ResNet18 or EfficientNet-B0) → feature vector."""
    def __init__(self, backbone='resnet18', out_dim=256, freeze=True):
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_f = base.fc.in_features; base.fc = nn.Identity()
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_f = base.classifier[1].in_features; base.classifier = nn.Identity()
        else:
            raise ValueError(backbone)
        self.encoder = base
        if freeze:
            for p in self.encoder.parameters(): p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(in_f, out_dim), nn.ReLU(), nn.Dropout(0.3))

    def unfreeze_top(self, n=2):
        for child in list(self.encoder.children())[-n:]:
            for p in child.parameters(): p.requires_grad = True
        tp = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [{self.backbone_name}] unfrozen top-{n}  trainable={tp:,}")

    def forward(self, x): return self.proj(self.encoder(x))


class CNNYieldModel(nn.Module):
    """CNN spatial features + county embedding + static soil → yield."""
    def __init__(self, backbone, static_feat, n_counties,
                 cnn_out=256, embed_dim=16, freeze=True):
        super().__init__()
        self.cnn          = CNNExtractor(backbone, cnn_out, freeze)
        self.county_embed = nn.Embedding(n_counties, embed_dim)  # EMBEDDING
        self.static_fc    = nn.Sequential(
            nn.Linear(static_feat, 32), nn.ReLU(), nn.Dropout(0.3))
        fuse = cnn_out + embed_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fuse,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64,1))

    def forward(self, img, x_static, county_ids):
        img  = RESIZE(img).to(DEVICE)
        cf   = self.cnn(img)
        emb  = self.county_embed(county_ids)
        st   = self.static_fc(x_static)
        return self.head(torch.cat([cf,emb,st],dim=1)).squeeze(-1)


class CNNDataset(Dataset):
    """Provides spatial image + static features per (county, year) sample."""
    def __init__(self, images, year_to_idx, df_years,
                 X_static, county_ids, y):
        self.images      = images
        self.img_idx     = torch.tensor(
            [year_to_idx.get(int(yr),0) for yr in df_years], dtype=torch.long)
        self.X_static    = torch.tensor(X_static,   dtype=torch.float32)
        self.county_ids  = torch.tensor(county_ids, dtype=torch.long)
        self.y           = torch.tensor(y,           dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.images[self.img_idx[i]], self.X_static[i], \
               self.county_ids[i], self.y[i]


# Build CNN datasets
gdd_col = sorted(c for c in DF.columns if c.startswith('gdd_'))[0]
IMAGES  = build_spatial_images(DF, gdd_col, grid_size=64)
years   = sorted(DF['year'].unique())
YEAR2IDX = {int(yr): i for i, yr in enumerate(years)}

split_year = years[int(len(years)*0.8)]
tr_mask_cnn = DF['year'].values < split_year
vl_mask_cnn = ~tr_mask_cnn

CNN_TR_DS = CNNDataset(IMAGES, YEAR2IDX, DF['year'].values[tr_mask_cnn],
                        Xst_tr, cid_tr, y_tr)
CNN_VL_DS = CNNDataset(IMAGES, YEAR2IDX, DF['year'].values[vl_mask_cnn],
                        Xst_vl, cid_vl, y_vl)

# Visualise spatial images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    ax.imshow(IMAGES[i].permute(1,2,0).numpy(), cmap='YlOrRd')
    ax.set_title(f'Year {years[i]}'); ax.axis('off')
plt.suptitle('Spatial Input Images (GDD → county grid)', fontsize=12)
plt.tight_layout(); plt.savefig('cnn_feature_maps.png', dpi=150); plt.show()


# ── CNN Hyperparameter search ─────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 2 — HYPERPARAMETER SEARCH  (ResNet18  lr × unfreeze_layers)")
print("="*60)

hp_cnn_results = []
for lr in [1e-3, 5e-4]:
    for unfreeze in [0, 2]:
        lbl = f'ResNet_lr{lr}_uf{unfreeze}'
        print(f"\n  lr={lr}  unfreeze_top={unfreeze}")
        m = CNNYieldModel('resnet18', STATIC_FEAT, N_COUNTIES, freeze=True).to(DEVICE)
        if unfreeze > 0: m.cnn.unfreeze_top(unfreeze)
        hist, m = run_training(m, CNN_TR_DS, CNN_VL_DS, PREP,
                                batch_size=64, epochs=25,
                                lr=lr, patience=7, label=lbl)
        hp_cnn_results.append({'lr': lr, 'unfreeze': unfreeze,
                                'best_R2':   round(max(hist['val_r2']),   4),
                                'best_RMSE': round(min(hist['val_rmse']), 4)})

HP_CNN_DF = pd.DataFrame(hp_cnn_results)
print("\n  CNN Hyperparameter Results:")
print(HP_CNN_DF.to_string(index=False))
best_cnn_row   = HP_CNN_DF.loc[HP_CNN_DF['best_R2'].idxmax()]
BEST_LR_CNN    = float(best_cnn_row['lr'])
BEST_UF_CNN    = int(best_cnn_row['unfreeze'])
print(f"\n  ✅ Best: lr={BEST_LR_CNN}  unfreeze={BEST_UF_CNN}")


# ── Compare CNN configs  (MODEL COMPARISON) ───────────────────────────────────
print("\n" + "="*60)
print("MODEL 2 — COMPARISON: ResNet18 vs EfficientNet-B0 (frozen / fine-tuned)")
print("="*60)

cnn_metrics   = {}
cnn_histories = {}

for backbone, freeze, lbl in [
    ('resnet18',       True,  'ResNet18-Frozen'),
    ('resnet18',       False, 'ResNet18-FineTuned'),
    ('efficientnet_b0',True,  'EffNet-Frozen'),
    ('efficientnet_b0',False, 'EffNet-FineTuned'),
]:
    print(f"\n  ── {lbl} ──")
    m = CNNYieldModel(backbone, STATIC_FEAT, N_COUNTIES, freeze=freeze).to(DEVICE)
    if not freeze: m.cnn.unfreeze_top(max(BEST_UF_CNN, 2))
    hist, m = run_training(m, CNN_TR_DS, CNN_VL_DS, PREP,
                            batch_size=64, epochs=40,
                            lr=BEST_LR_CNN, patience=10, label=lbl)
    cnn_histories[lbl] = hist
    cnn_metrics[lbl]   = final_eval(m, CNN_VL_DS, PREP, label=lbl)


# ─── CELL 9 : MODEL 2 PLOTS ──────────────────────────────────────────────────
palette4 = ['#e74c3c','#e67e22','#2ecc71','#3498db']
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for (lbl, h), col in zip(cnn_histories.items(), palette4):
    axes[0].plot(h['val_loss'], color=col, label=lbl)
    axes[1].plot(h['val_r2'],   color=col, label=lbl)
axes[0].set(title='CNN Loss',      xlabel='Epoch', ylabel='MSE'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].set(title='CNN Val R²',    xlabel='Epoch', ylabel='R²');  axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
plt.suptitle('MODEL 2: CNN (Frozen vs Fine-Tuned)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('cnn_results.png', dpi=150); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
lbls4 = list(cnn_metrics.keys())
for ax, metric in zip(axes, ['RMSE','MAE','R2','MAPE']):
    vals = [cnn_metrics[l][metric] for l in lbls4]
    bars = ax.bar(range(len(lbls4)), vals, color=palette4, edgecolor='k')
    ax.set_xticks(range(len(lbls4)))
    ax.set_xticklabels(lbls4, rotation=20, ha='right', fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.001,
                f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_title(metric); ax.grid(axis='y', alpha=.3)
plt.suptitle('CNN Model Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('cnn_comparison.png', dpi=150, bbox_inches='tight'); plt.show()


# ─── CELL 10 : MODEL 3 — TRANSFORMER ─────────────────────────────────────────
# ===========================================================================

class PositionalEncoding(nn.Module):
    """
    Sine/cosine positional encoding so the Transformer knows month order.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, d_model, max_len=12, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()
                        * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, T, d)

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerYieldModel(nn.Module):
    """
    Temporal Transformer with:
      • Multi-head self-attention  (which month→month relationships matter?)
      • Positional encoding        (month order — EMBEDDING USAGE)
      • County embedding           (spatial identity — EMBEDDING USAGE)
      • Static soil branch         (fused into prediction head)
    """
    def __init__(self, seq_feat, static_feat, n_counties,
                 d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, embed_dim=16, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.input_proj = nn.Linear(seq_feat, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True)
        self.transformer  = nn.TransformerEncoder(enc_layer, n_layers)
        self.county_embed = nn.Embedding(n_counties, embed_dim)   # EMBEDDING
        self.static_fc    = nn.Sequential(
            nn.Linear(static_feat,32), nn.ReLU(), nn.Dropout(dropout))
        fuse = d_model + embed_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fuse,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),  nn.ReLU(), nn.Linear(64,1))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x_seq, x_static, county_ids):
        x      = self.pos_enc(self.input_proj(x_seq))  # (B,T,d)
        enc    = self.transformer(x)                    # (B,T,d)
        pooled = enc.mean(dim=1)                        # (B,d)
        emb    = self.county_embed(county_ids)
        st     = self.static_fc(x_static)
        return self.head(torch.cat([pooled,emb,st],dim=1)).squeeze(-1)

    def attention_map(self, x_seq, x_static, county_ids):
        """Return (T×T) self-attention from first encoder layer."""
        x = self.pos_enc(self.input_proj(x_seq))
        with torch.no_grad():
            _, attn = self.transformer.layers[0].self_attn(
                x, x, x, average_attn_weights=True)
        return attn   # (B,T,T)


# ── Transformer Hyperparameter search ────────────────────────────────────────
print("\n" + "="*60)
print("MODEL 3 — HYPERPARAMETER SEARCH  (Transformer  d_model × n_heads)")
print("="*60)

hp_tf_results = []
for d_model in [32, 64]:
    for n_heads in [2, 4]:
        if d_model % n_heads != 0: continue
        lbl = f'TF_d{d_model}_h{n_heads}'
        print(f"\n  d_model={d_model}  n_heads={n_heads}")
        m = TransformerYieldModel(SEQ_FEAT, STATIC_FEAT, N_COUNTIES,
                                   d_model=d_model, n_heads=n_heads).to(DEVICE)
        hist, m = run_training(m, TR_DS, VL_DS, PREP,
                                epochs=30, lr=1e-3, patience=7,
                                label=lbl, optimizer_cls=optim.AdamW,
                                scheduler='cosine')
        hp_tf_results.append({'d_model': d_model, 'n_heads': n_heads,
                               'best_R2':   round(max(hist['val_r2']),   4),
                               'best_RMSE': round(min(hist['val_rmse']), 4)})

HP_TF_DF = pd.DataFrame(hp_tf_results)
print("\n  Transformer Hyperparameter Results:")
print(HP_TF_DF.to_string(index=False))
best_tf_row = HP_TF_DF.loc[HP_TF_DF['best_R2'].idxmax()]
BEST_D      = int(best_tf_row['d_model'])
BEST_H_TF   = int(best_tf_row['n_heads'])
print(f"\n  ✅ Best: d_model={BEST_D}  n_heads={BEST_H_TF}")


# ── Compare Transformer variants  (MODEL COMPARISON) ─────────────────────────
print("\n" + "="*60)
print("MODEL 3 — COMPARISON: TF-Small  vs  TF-Base  vs  TF-Deep")
print("="*60)

tf_metrics   = {}
tf_histories = {}

for lbl, n_layers in [('TF-Small',1), ('TF-Base',2), ('TF-Deep',4)]:
    print(f"\n  ── {lbl}  (layers={n_layers}) ──")
    m = TransformerYieldModel(SEQ_FEAT, STATIC_FEAT, N_COUNTIES,
                               d_model=BEST_D, n_heads=BEST_H_TF,
                               n_layers=n_layers).to(DEVICE)
    hist, m = run_training(m, TR_DS, VL_DS, PREP,
                            epochs=60, lr=1e-3, patience=10,
                            label=lbl, optimizer_cls=optim.AdamW,
                            scheduler='cosine')
    tf_histories[lbl] = hist
    tf_metrics[lbl]   = final_eval(m, VL_DS, PREP, label=lbl)

    if lbl == 'TF-Base':
        BEST_TF_MODEL = m


# ─── CELL 11 : MODEL 3 PLOTS ─────────────────────────────────────────────────
palette3 = ['#9b59b6','#e74c3c','#3498db']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for (lbl, h), col in zip(tf_histories.items(), palette3):
    axes[0].plot(h['val_loss'], color=col, label=lbl)
    axes[1].plot(h['val_r2'],   color=col, label=lbl)
axes[0].set(title='Transformer Loss', xlabel='Epoch', ylabel='MSE'); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].set(title='Transformer Val R²', xlabel='Epoch', ylabel='R²'); axes[1].legend(); axes[1].grid(alpha=.3)
plt.suptitle('MODEL 3: Transformer Variants', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('transformer_results.png', dpi=150); plt.show()

# Self-attention heatmap
BEST_TF_MODEL.eval()
xs, xst, cids, _ = next(iter(DataLoader(VL_DS, batch_size=256)))
attn_map = BEST_TF_MODEL.attention_map(
    xs.to(DEVICE), xst.to(DEVICE), cids.to(DEVICE))
avg_attn_map = attn_map.mean(0).cpu().numpy()
plt.figure(figsize=(7,6))
plt.imshow(avg_attn_map, cmap='Blues', aspect='auto')
plt.colorbar(label='Attention Weight')
T_ = len(PREP.month_labels)
plt.xticks(range(T_), PREP.month_labels, rotation=45)
plt.yticks(range(T_), PREP.month_labels)
plt.title('Transformer Self-Attention Map\n(Month → Month dependencies)', fontsize=12)
plt.tight_layout(); plt.savefig('transformer_attention.png', dpi=150); plt.show()


# ─── CELL 12 : FINAL CROSS-ARCHITECTURE EVALUATION ───────────────────────────
print("\n" + "█"*60)
print("  FINAL EVALUATION — All Models")
print("█"*60)


def best_variant(d):
    """Pick variant with highest R² from a metrics dict."""
    return max(d.items(), key=lambda kv: kv[1]['R2'])


# Build summary table
summary_rows = []
for arch, d in [('LSTM/RNN/GRU', lstm_metrics),
                ('CNN',           cnn_metrics),
                ('Transformer',   tf_metrics)]:
    name, m = best_variant(d)
    summary_rows.append({
        'Architecture': arch, 'Best Variant': name,
        'RMSE': round(m['RMSE'],4), 'MAE':  round(m['MAE'],4),
        'R²':   round(m['R2'],4),   'MAPE %': round(m['MAPE'],2)})

SUMMARY = pd.DataFrame(summary_rows).set_index('Architecture')
print("\n" + SUMMARY.to_string())
SUMMARY.to_csv('final_summary.csv')


# ── Bar chart: all architectures ─────────────────────────────────────────────
lbls_final  = [r['Best Variant'] for r in summary_rows]
arch_labels = [r['Architecture'] for r in summary_rows]
colors_final= ['#2ecc71','#3498db','#9b59b6']
metric_info = [('RMSE','RMSE (↓ better)', False),
               ('MAE', 'MAE  (↓ better)', False),
               ('R²',  'R²   (↑ better)', True),
               ('MAPE %','MAPE % (↓ better)', False)]

fig = plt.figure(figsize=(18,6))
gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)
for idx, (key, title, hb) in enumerate(metric_info):
    ax   = fig.add_subplot(gs[idx])
    vals = list(SUMMARY[key])
    bars = ax.bar(arch_labels, vals, color=colors_final, edgecolor='k')
    best_i = (np.argmax if hb else np.argmin)(vals)
    for i, (b, v) in enumerate(zip(bars, vals)):
        col = 'darkgreen' if i == best_i else 'black'
        ax.text(b.get_x()+b.get_width()/2,
                b.get_height()+max(vals)*0.01,
                f'{v:.4f}', ha='center', fontsize=9,
                fontweight='bold', color=col)
    bars[best_i].set_edgecolor('gold'); bars[best_i].set_linewidth(2.5)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticklabels(arch_labels, rotation=10)
    ax.grid(axis='y', alpha=.3)
plt.suptitle('Final Comparison — LSTM/RNN/GRU  vs  CNN  vs  Transformer',
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight'); plt.show()


# ── Radar / Spider chart ──────────────────────────────────────────────────────
df_radar = SUMMARY[['RMSE','MAE','R²','MAPE %']].copy()
df_radar['RMSE']   = 1/(1+df_radar['RMSE'])
df_radar['MAE']    = 1/(1+df_radar['MAE'])
df_radar['MAPE %'] = 1/(1+df_radar['MAPE %'])
metrics_r = list(df_radar.columns)
N = len(metrics_r)
angles = [n/N*2*math.pi for n in range(N)] + [0]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
for (arch, row), col in zip(df_radar.iterrows(), colors_final):
    vals = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, vals, color=col, linewidth=2, label=arch)
    ax.fill(angles, vals, color=col, alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_r, fontsize=12)
ax.set_yticks([]); ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1))
ax.set_title('Radar Chart — All higher = better', fontsize=12, pad=20, fontweight='bold')
plt.tight_layout(); plt.savefig('radar_comparison.png', dpi=150, bbox_inches='tight'); plt.show()


# ── Heatmap of ALL variants ───────────────────────────────────────────────────
all_rows_hm = {}
for prefix, d in [('LSTM', lstm_metrics),
                  ('CNN',  cnn_metrics),
                  ('TF',   tf_metrics)]:
    for name, m in d.items():
        all_rows_hm[name] = m

HM_DF = pd.DataFrame(all_rows_hm).T[['RMSE','MAE','R2','MAPE']]
fig, ax = plt.subplots(figsize=(10, max(5, len(HM_DF)*0.55+1)))
im = ax.imshow(HM_DF.values.astype(float), cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(HM_DF.columns))); ax.set_xticklabels(HM_DF.columns, fontsize=12)
ax.set_yticks(range(len(HM_DF.index)));   ax.set_yticklabels(HM_DF.index, fontsize=9)
for i in range(len(HM_DF.index)):
    for j in range(len(HM_DF.columns)):
        ax.text(j, i, f'{HM_DF.values[i,j]:.3f}',
                ha='center', va='center', fontsize=8, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.02)
ax.set_title('All Model Variants — Evaluation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('metrics_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()


# ── Print winner ──────────────────────────────────────────────────────────────
best_arch = SUMMARY['R²'].idxmax()
print("\n" + "★"*60)
print(f"  🏆 Best Architecture : {best_arch}")
print(f"     Best Variant     : {SUMMARY.loc[best_arch,'Best Variant']}")
print(f"     R²               : {SUMMARY.loc[best_arch,'R²']:.4f}")
print(f"     RMSE             : {SUMMARY.loc[best_arch,'RMSE']:.4f}")
print(f"     MAE              : {SUMMARY.loc[best_arch,'MAE']:.4f}")
print("★"*60)

print("\n📁 Saved files:")
for f in ['final_summary.csv','final_comparison.png','radar_comparison.png',
          'metrics_heatmap.png','lstm_results.png','lstm_attention.png',
          'cnn_feature_maps.png','cnn_results.png','cnn_comparison.png',
          'transformer_results.png','transformer_attention.png']:
    print(f"   • {f}")