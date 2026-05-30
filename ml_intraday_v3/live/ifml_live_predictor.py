"""
IFML Live Predictor — Real-time microstructure inference for live trading.

Wraps the trained informed_flow_ml model for deployment.
At each 5-min bar close, receives tick data (or OHLCV fallback),
computes microstructure features, and returns a trade signal.

Two modes:
  TICK mode (preferred): feed raw tick DataFrame at bar close
    → computes full microstructure features (OFI, Kyle's lambda, etc.)
  OHLCV mode (fallback): feed just OHLCV bar + volume
    → micro features default to 0 (neutral), OHLCV features still active

Usage:
    predictor = IFMLLivePredictor.load("ml_intraday_v3/models/informed_flow_ml_long.pkl")
    result = predictor.predict_bar(bar_ohlcv, tick_df=ticks)  # or tick_df=None
"""

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE  = Path(__file__).parents[2]
MODELS = BASE / "ml_intraday_v3" / "models"

LARGE_TRADE_THRESH = 5  # contracts — matches build_microstructure_features.py


class IFMLLivePredictor:
    def __init__(self, bundle: dict):
        self.model       = bundle["model"]
        self.calibrator  = bundle["calibrator"]
        self.features    = bundle["features"]
        self.micro_feats = bundle.get("micro_features", [])
        self.ohlcv_feats = bundle.get("ohlcv_features", [])
        self.cfg         = bundle["cfg"]
        self.combine_cfg = bundle["combine_cfg"]

        # Rolling history for OHLCV-derived feature computation
        # Keep last 80 bars (> longest rolling window = 78)
        self._history: deque = deque(maxlen=80)

        # Normalization state for microstructure (rolling z-score)
        self._micro_history: deque = deque(maxlen=80)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IFMLLivePredictor":
        if path is None:
            path = MODELS / "informed_flow_ml_long.pkl"
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle)

    def feed_bar(
        self,
        bar: dict,
        tick_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Feed one completed 5-min bar and get a prediction.

        bar: dict with keys: timestamp, open, high, low, close, volume (or total_vol)
        tick_df: optional DataFrame of raw ticks for this bar with columns:
                 [ts_recv (ns int), price (float), size (float), side ('B'/'A'/'N')]

        Returns dict:
          p_long: calibrated probability of profitable long entry
          signal: True if p_long >= threshold
          n_contracts: recommended position size (0 if no signal)
          micro_available: whether tick data was used
          features_used: dict of feature values fed to model
        """
        micro_raw = self._compute_micro_features(tick_df)
        bar_row   = {**bar, **micro_raw}
        self._history.append(bar_row)
        self._micro_history.append(micro_raw)

        if len(self._history) < 25:
            return {"p_long": 0.0, "signal": False, "n_contracts": 0,
                    "micro_available": tick_df is not None, "reason": "warming_up"}

        hist_df = pd.DataFrame(list(self._history))
        hist_df.index = pd.to_datetime([r["timestamp"] for r in self._history], utc=True)

        feat_row = self._build_feature_row(hist_df)
        if feat_row is None:
            return {"p_long": 0.0, "signal": False, "n_contracts": 0,
                    "micro_available": tick_df is not None, "reason": "feature_computation_failed"}

        X = feat_row.values.reshape(1, -1).astype(np.float32)
        raw_p = self.model.predict_proba(X)[0, 1]
        cal_p = float(self.calibrator.predict([raw_p])[0])

        signal    = cal_p >= self.cfg["long_threshold"]
        n_contr   = self._default_contracts(cal_p) if signal else 0

        return {
            "p_long":          round(cal_p, 4),
            "signal":          signal,
            "n_contracts":     n_contr,
            "micro_available": tick_df is not None,
            "threshold_used":  self.cfg["long_threshold"],
            "features_used":   dict(zip(self.features, X[0].tolist())),
        }

    def _compute_micro_features(self, tick_df: Optional[pd.DataFrame]) -> dict:
        """Compute microstructure features from tick data, or return neutral defaults."""
        if tick_df is None or tick_df.empty:
            return {f: np.nan for f in self.micro_feats}

        price = tick_df["price"].values.astype(float)
        size  = tick_df["size"].values.astype(float)
        side  = tick_df["side"].values
        ts_ns = tick_df["ts_recv"].values.astype(np.int64)

        buy_mask  = side == "B"
        sell_mask = side == "A"
        large_mask = size >= LARGE_TRADE_THRESH

        buy_vol  = size[buy_mask].sum()
        sell_vol = size[sell_mask].sum()
        total_vol = size.sum()
        ofi = buy_vol - sell_vol

        lg_buy  = size[buy_mask  & large_mask].sum()
        lg_sell = size[sell_mask & large_mask].sum()
        sm_buy  = size[buy_mask  & ~large_mask].sum()
        sm_sell = size[sell_mask & ~large_mask].sum()
        lg_vol  = size[large_mask].sum()
        sm_vol  = size[~large_mask].sum()

        lg_ofi = lg_buy - lg_sell
        sm_ofi = sm_buy - sm_sell
        bar_ret = price[-1] - price[0] if len(price) > 0 else 0.0

        denom = abs(ofi) if abs(ofi) > 0 else np.nan
        kyles_lambda = abs(bar_ret) / denom if not np.isnan(denom or np.nan) else np.nan

        dp = np.diff(price)
        roll_spread = np.nan
        if len(dp) >= 2:
            cov = np.cov(dp[:-1], dp[1:])[0, 1]
            roll_spread = 2 * np.sqrt(max(-cov, 0))

        # Sub-bar OFI (first 10% vs last 10% of bar)
        ofi_early = ofi_late = ofi_accel = 0.0
        if len(ts_ns) > 1 and ts_ns[-1] > ts_ns[0]:
            dur = ts_ns[-1] - ts_ns[0]
            early_cut = ts_ns[0] + dur * 0.1
            late_start = ts_ns[0] + dur * 0.9
            e_mask = ts_ns <= early_cut
            l_mask = ts_ns >= late_start
            ofi_early = float(size[e_mask & buy_mask].sum() - size[e_mask & sell_mask].sum())
            ofi_late  = float(size[l_mask & buy_mask].sum() - size[l_mask & sell_mask].sum())
            ofi_accel = ofi_late - ofi_early

        # Longest same-side run
        max_run = 1; cur_run = 1
        for i in range(1, len(side)):
            if side[i] == side[i-1] and side[i] in ("B", "A"):
                cur_run += 1; max_run = max(max_run, cur_run)
            else:
                cur_run = 1

        lg_sm_diverge = float((lg_ofi > 0) != (sm_ofi > 0)) if (lg_vol > 0 and sm_vol > 0) else 0.0

        return {
            "ofi_imb":      ofi / total_vol if total_vol > 0 else 0.0,
            "lg_ofi_imb":   lg_ofi / lg_vol if lg_vol > 0 else 0.0,
            "lg_sm_diverge": lg_sm_diverge,
            "large_frac":   lg_vol / total_vol if total_vol > 0 else 0.0,
            "ofi_accel":    ofi_accel,
            "ofi_early":    ofi_early,
            "ofi_late":     ofi_late,
            "kyles_lambda": kyles_lambda,
            "roll_spread":  roll_spread,
            "trade_rate":   len(tick_df) / 300.0,
            "max_run":      float(max_run),
            "avg_size":     float(np.mean(size)),
            "max_size":     float(np.max(size)),
            "size_std":     float(np.std(size)) if len(size) > 1 else 0.0,
        }

    def _build_feature_row(self, hist_df: pd.DataFrame) -> Optional[pd.Series]:
        """Build the full feature vector from rolling history."""
        try:
            c   = hist_df["close"]
            atr = self._compute_atr(hist_df)
            last_atr = float(atr.iloc[-1])
            if np.isnan(last_atr) or last_atr <= 0:
                return None

            features = {}

            # OHLCV
            for n in [1, 3, 6]:
                features[f"ret_{n}"] = float(np.log(c.iloc[-1] / c.iloc[-(n+1)])) if len(c) > n else 0.0

            rsi14 = self._rsi(c, 14)
            features["rsi_14"] = float(rsi14.iloc[-1]) if not np.isnan(rsi14.iloc[-1]) else 50.0

            ema9  = c.ewm(span=9,  min_periods=9).mean()
            ema21 = c.ewm(span=21, min_periods=21).mean()
            features["ema9_ratio"]  = float((c.iloc[-1] / ema9.iloc[-1] - 1) * 100)
            features["ema21_ratio"] = float((c.iloc[-1] / ema21.iloc[-1] - 1) * 100)

            hi = hist_df["high"].iloc[-1]
            lo = hist_df["low"].iloc[-1]
            op = hist_df["open"].iloc[-1]
            features["norm_range"] = float((hi - lo) / last_atr)
            features["norm_body"]  = float((c.iloc[-1] - op) / last_atr)

            tv = hist_df.get("total_vol", pd.Series(np.nan, index=hist_df.index))
            tv = tv.replace(0, np.nan)
            if tv.notna().sum() > 20:
                features["vol_z"] = float((tv.iloc[-1] - tv.rolling(20).mean().iloc[-1]) /
                                          max(tv.rolling(20).std().iloc[-1], 1e-9))
            else:
                features["vol_z"] = 0.0

            if atr.rolling(20).std().iloc[-1] > 0:
                features["atr_z"] = float((last_atr - atr.rolling(20).mean().iloc[-1]) /
                                          atr.rolling(20).std().iloc[-1])
            else:
                features["atr_z"] = 0.0

            if "vwap" in hist_df.columns:
                features["vwap_dev"] = float((c.iloc[-1] - hist_df["vwap"].iloc[-1]) / last_atr)
            else:
                features["vwap_dev"] = 0.0

            ts = hist_df.index[-1]
            h_et = (ts.hour - 5) % 24
            features["hour_sin"] = float(np.sin(2 * np.pi * h_et / 24))
            features["hour_cos"] = float(np.cos(2 * np.pi * h_et / 24))
            features["dow"]      = float(ts.dayofweek)

            # Microstructure (with rolling normalization)
            window = min(len(self._micro_history), 40)
            micro_keys = ["ofi_imb", "lg_ofi_imb", "ofi_accel", "ofi_early", "ofi_late",
                          "kyles_lambda", "roll_spread", "trade_rate", "avg_size", "max_size",
                          "size_std", "max_run", "lg_sm_diverge", "large_frac"]

            for k in micro_keys:
                vals = [h.get(k, np.nan) for h in list(self._micro_history)[-window:]]
                vals_arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
                cur_val = self._micro_history[-1].get(k, np.nan)
                if len(vals_arr) >= 10 and not np.isnan(cur_val):
                    mu = vals_arr.mean()
                    sd = vals_arr.std()
                    features[k] = float((cur_val - mu) / max(sd, 1e-9))
                else:
                    features[k] = 0.0  # neutral default when not enough history

            row = pd.Series({f: features.get(f, 0.0) for f in self.features})
            return row

        except Exception as e:
            print(f"Feature computation error: {e}")
            return None

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"]  - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, min_periods=period).mean()

    @staticmethod
    def _rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
        l = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    def _default_contracts(self, p_long: float) -> int:
        base = self.cfg["base_contracts"]
        max_c = self.cfg["max_contracts"]
        thresh = self.cfg["long_threshold"]
        above = (p_long - thresh) / max(1 - thresh, 0.01)
        return min(base + int(above * 1.5), max_c)

    def reset(self):
        """Call at start of each trading session."""
        self._history.clear()
        self._micro_history.clear()
