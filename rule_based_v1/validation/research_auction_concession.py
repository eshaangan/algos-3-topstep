"""Treasury auction concession & reversal -- mechanism screen with matched control.

H1 CONCESSION  price FALLS over the `pre` minutes ending at auction close
H2 REVERSAL    price RISES over the `post` minutes after the result release

The statistic that counts is EVENT MINUS MATCHED CONTROL. The control is the
same clock window on weekdays carrying NO coupon auction of ANY tenor -- using
only the tested tenor would leave 3y/30y/2y refunding days inside the control.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ET = "America/New_York"

def load_bars(p):
    b = pd.read_parquet(p)
    if b.index.tz is None: raise ValueError("naive index; refuse to guess tz")
    b = b.tz_convert(ET).sort_index()
    return b[~b.index.duplicated(keep="last")]

def assert_halt(bars):
    """CME halt is 16:00-17:00 Chicago all year => first bar back is 18:0x ET
    in BOTH seasons. A tape localized as fixed -05:00 fails this in summer."""
    g = bars.index.to_series().diff().dt.total_seconds()/60.0
    at = pd.Series(g[(g > 50) & (g < 80)].index)
    out = {}
    for name, months in (("winter",[1,2,12]),("summer",[6,7,8])):
        vc = at[at.dt.month.isin(months)].dt.hour.value_counts()
        out[name] = vc.to_dict()
        if vc.empty or vc.idxmax() != 18:
            raise AssertionError(f"halt returns at hour {None if vc.empty else vc.idxmax()} in {name}, expected 18 ET: {out}")
    return out

def px(bars, ts, tol=15):
    i = bars.index.searchsorted(ts, side="right") - 1
    if i < 0: return None
    if (ts - bars.index[i]) > pd.Timedelta(minutes=tol): return None
    return float(bars["close"].iloc[i])

def ret(bars, a, b):
    x, y = px(bars, a), px(bars, b)
    return None if (x is None or y is None) else y - x

def collect(bars, anchors, pre, post, buf):
    rows = []
    for a in anchors.itertuples(index=False):
        c, r = a.auction_close, a.result_release
        p = ret(bars, c - pd.Timedelta(minutes=pre), c)
        q = ret(bars, r + pd.Timedelta(minutes=buf), r + pd.Timedelta(minutes=buf+post))
        if p is None or q is None: continue
        rows.append({"day": c.date(), "year": c.year, "term": getattr(a,"security_term","control"),
                     "pre_points": p, "post_points": q})
    return pd.DataFrame(rows)

def tstat(x):
    x = np.asarray(x,float)
    return float("nan") if len(x)<2 or x.std(ddof=1)==0 else float(x.mean()/(x.std(ddof=1)/np.sqrt(len(x))))

def welch(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    d = a.mean()-b.mean()
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    return float(d), (float(d/se) if se>0 else float("nan"))

def gini(x):
    x = np.sort(np.asarray(x,float)); n=len(x)
    if n==0: return float("nan")
    x = x - x.min(); s = x.sum()
    return 0.0 if s==0 else float((2*np.arange(1,n+1)-n-1).dot(x)/(n*s))

def leg(name, ev, ct, pv, cost):
    e, c = ev*pv, ct*pv
    d, td = welch(e, c)
    side = -1.0 if name=="concession" else 1.0
    net = side*e - cost
    mabs = float(np.abs(e).mean())
    return {"leg":name,"n_event":len(e),"n_control":len(c),
            "event_mean_$":round(float(e.mean()),2),"event_t":round(tstat(e),3),
            "control_mean_$":round(float(c.mean()),2),"control_t":round(tstat(c),3),
            "event_minus_control_$":round(d,2),"event_minus_control_t":round(td,3),
            "mean_abs_move_$":round(mabs,2),"cost_ratio":round(cost/mabs,4) if mabs else None,
            "net_mean_$":round(float(net.mean()),2),"net_t":round(tstat(net),3),
            "net_win_rate":round(float((net>0).mean()),4),"gini":round(gini(net),3)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars",type=Path,required=True)
    ap.add_argument("--auctions",type=Path,required=True)
    ap.add_argument("--term",action="append",default=None)
    ap.add_argument("--pre",type=int,default=180)
    ap.add_argument("--post",type=int,default=180)
    ap.add_argument("--buffer",type=int,default=5)
    ap.add_argument("--point-value",type=float,default=1000.0)
    ap.add_argument("--tick-value",type=float,default=15.625)
    ap.add_argument("--commission",type=float,default=1.00)
    ap.add_argument("--slip-ticks",type=float,default=0.5)
    ap.add_argument("--dev-end",default="2023-12-31")
    ap.add_argument("--out",type=Path,required=True)
    a = ap.parse_args()

    bars = load_bars(a.bars); halt = assert_halt(bars)
    cal = pd.read_csv(a.auctions)
    for col in ("auction_close","result_release"):
        cal[col] = pd.to_datetime(cal[col], utc=True).dt.tz_convert(ET)
    cal["day"] = cal["auction_close"].dt.date
    cal = cal[(cal["auction_close"] >= bars.index.min()) & (cal["auction_close"] <= bars.index.max())]
    all_auction_days = set(cal["day"])                       # every tenor -> clean control
    sel = cal if not a.term else cal[cal["security_term"].isin(a.term)]

    cost = 2.0*(a.commission + a.slip_ticks*a.tick_value)

    events = collect(bars, sel, a.pre, a.post, a.buffer)

    # control: same clock, weekdays with NO coupon auction of any tenor
    mode_close = sel["auction_close"].dt.time.mode().iat[0]
    mode_rel   = sel["result_release"].dt.time.mode().iat[0]
    days = sorted({d.date() for d in bars.index})
    lo, hi = min(sel["day"]), max(sel["day"])
    cdays = [d for d in days if lo <= d <= hi and d.weekday()<5 and d not in all_auction_days]
    ctrl_anchor = pd.DataFrame({
        "auction_close":[pd.Timestamp.combine(d,mode_close).tz_localize(ET) for d in cdays],
        "result_release":[pd.Timestamp.combine(d,mode_rel).tz_localize(ET) for d in cdays],
        "security_term":"control"})
    control = collect(bars, ctrl_anchor, a.pre, a.post, a.buffer)

    out = {"bars":str(a.bars),"halt_return_hour_et":halt,"auctions":str(a.auctions),
           "terms":a.term or "ALL_NOMINAL_COUPON",
           "windows":{"pre_min":a.pre,"post_min":a.post,"buffer_min":a.buffer},
           "cost":{"commission_per_side":a.commission,"slip_ticks_per_side":a.slip_ticks,
                   "tick_value":a.tick_value,"round_turn_$":round(cost,2)},
           "n_events":len(events),"n_control":len(control),
           "legs":[],"eras":{},"by_year":{}}
    pairs = (("concession","pre_points"),("reversal","post_points"))
    for nm,col in pairs:
        out["legs"].append(leg(nm, events[col].to_numpy(), control[col].to_numpy(), a.point_value, cost))
    de = pd.Timestamp(a.dev_end).date()
    for era, me, mc in (("dev",events["day"]<=de,control["day"]<=de),("val",events["day"]>de,control["day"]>de)):
        out["eras"][era] = [leg(nm, events.loc[me,col].to_numpy(), control.loc[mc,col].to_numpy(), a.point_value, cost) for nm,col in pairs]
    for y,g in events.groupby("year"):
        out["by_year"][int(y)] = {"n":len(g),
            "concession_$":round(float(g["pre_points"].mean()*a.point_value),2),
            "reversal_$":round(float(g["post_points"].mean()*a.point_value),2)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2))
    events.to_csv(a.out.with_suffix(".events.csv"),index=False)
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
