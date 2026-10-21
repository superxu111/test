"""r4_e35_lanl_placement.py — E35：LANL 真实红队事件上的放置重加权评估（⑦a，零训练）。

目的（review-panel ⑦a）：把 E32 的放置族分析搬到 LANL 真实红队数据上，检验
「攻击者可控构成改变池化结论」是否在真实企业认证数据上复现。

口径与限制（必须随结果一并披露）：
  * 这是「放置重加权」（placement reweighting）——对已观测红队正例按可见性层重新加权，
    不是「重新植入攻击」（LANL 事件不可移动）。因此它回答：若攻击者能把活动集中到
    其偏好的可见性层，池化结论会怎样变化；不回答植入可行性。
  * 分层：日内 cross 边（恰一端属于遮蔽集 S）按度分位分 K=5 层。
  * 三个口径与 E32 代码逐位一致：
      pooled(w) = Σ_s w_s Σ_{i∈pos_s} #(neg < a_i) / (Σ_s w_s |pos_s| · |neg|)   （含跨层项）
      std       = 占据层内 AUC 的算术平均（E32 代码口径，非 benign-occupancy 加权）
      F         = 占据层内 AUC 的最小值
  * 放置族：w_s ∝ mean_{pos∈s}(deg+1)^alpha，alpha ∈ {-2,-1,-0.5,0,0.25,0.5,0.75,1.0,1.5}；
    alpha=0 为均匀放置。与会话 E32/K=5 同族但独立实现、独立数据。
  * 只读 results/r2/lanl_blindness/scores_e11_*.npz；输出到新目录 results/r4/e35_lanl_placement/。

运行：python scripts/r4_e35_lanl_placement.py
"""
import json, os, itertools
import numpy as np

SRC = '/root/autodl-tmp/results/r2/lanl_blindness'
OUT = '/root/autodl-tmp/results/r4/e35_lanl_placement'
os.makedirs(OUT, exist_ok=True)

ARMS = ['full', 'blind5', 'blind0']
SEEDS = [0, 1, 2]
DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
K = 5
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def ranknorm(v):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(v, kind='mergesort')
    r = np.empty(n)
    r[order] = np.arange(n, dtype=np.float64)
    # average ties
    s = v[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r / (n - 1)


def counts_lt(pos, neg_sorted):
    """return #(neg < p), #(neg == p) for each p via searchsorted."""
    lo = np.searchsorted(neg_sorted, pos, side='left')
    hi = np.searchsorted(neg_sorted, pos, side='right')
    return lo, hi - lo


def auc_stratum(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    ns = np.sort(neg)
    lo, eq = counts_lt(pos, ns)
    wins = lo + 0.5 * eq
    return float(wins.mean() / len(ns))


def day_metrics(z, arm, seed, day, alpha):
    pu = f'{day}_cross_u'
    if pu not in z:
        return None
    y = z[f'{day}_cross_y'].astype(int)
    deg = z[f'{day}_cross_deg'].astype(np.float64)
    a = z[f'asimpl__{day}_cross_a'].astype(np.float64)
    # sanity: higher score = more anomalous
    q = np.clip((ranknorm(deg) * K).astype(int), 0, K - 1)
    strata = {}
    for s in range(K):
        m = q == s
        pos = a[m & (y == 1)]
        neg = a[m & (y == 0)]
        au = auc_stratum(pos, neg)
        if au is None:
            continue
        w = float(np.mean((deg[m & (y == 1)] + 1.0) ** alpha)) if (m & (y == 1)).any() else 0.0
        strata[s] = {'auc': au, 'w': w, 'n_pos': int((m & (y == 1)).sum()),
                     'n_neg': int((m & (y == 0)).sum())}
    if not strata:
        return None
    # pooled with weighted positives, all negatives pooled
    neg_all = np.sort(a[y == 0])
    num = 0.0
    den = 0.0
    for s, d in strata.items():
        m = (q == s) & (y == 1)
        pos = a[m]
        lo, eq = counts_lt(pos, neg_all)
        wins = lo + 0.5 * eq
        num += d['w'] * float(wins.sum())
        den += d['w'] * len(pos) * len(neg_all)
    pooled = num / den if den > 0 else None
    std = float(np.mean([d['auc'] for d in strata.values()]))
    F = float(min(d['auc'] for d in strata.values()))
    return {'pooled': pooled, 'std': std, 'F': F,
            'n_pos': int(y.sum()), 'n_neg': int((y == 0).sum()),
            'occupied': len(strata),
            'strata': {str(s): d for s, d in strata.items()}}


def main():
    rows = []
    for arm, seed in itertools.product(ARMS, SEEDS):
        f = f'{SRC}/scores_e11_{arm}_seed{seed}.npz'
        if not os.path.exists(f):
            print('missing', f); continue
        z = np.load(f, allow_pickle=True)
        for alpha in ALPHAS:
            per_day = []
            for day in DAYS:
                m = day_metrics(z, arm, seed, day, alpha)
                if m:
                    per_day.append(m)
            if not per_day:
                continue
            rows.append({
                'arm': arm, 'seed': seed, 'alpha': alpha,
                'n_days': len(per_day),
                'pooled': float(np.mean([m['pooled'] for m in per_day])),
                'std': float(np.mean([m['std'] for m in per_day])),
                'F': float(np.mean([m['F'] for m in per_day])),
                'mean_occupied': float(np.mean([m['occupied'] for m in per_day])),
                'n_pos_total': int(sum(m['n_pos'] for m in per_day)),
                'per_day': [{'day': d, **m} for d, m in zip(DAYS, per_day)],
            })
        print('done', arm, seed, flush=True)

    with open(f'{OUT}/results_e35.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')

    # summary: per arm, per alpha -> mean / min / max over seeds
    summ = {}
    for arm in ARMS:
        summ[arm] = {}
        for alpha in ALPHAS:
            xs = [r for r in rows if r['arm'] == arm and r['alpha'] == alpha]
            if not xs:
                continue
            summ[arm][str(alpha)] = {
                'pooled_mean': float(np.mean([x['pooled'] for x in xs])),
                'pooled_per_seed': [round(x['pooled'], 4) for x in xs],
                'std_mean': float(np.mean([x['std'] for x in xs])),
                'F_mean': float(np.mean([x['F'] for x in xs])),
                'mean_occupied': float(np.mean([x['mean_occupied'] for x in xs])),
            }
    with open(f'{OUT}/analysis_e35.json', 'w') as fh:
        json.dump({
            'experiment': 'E35',
            'purpose': 'LANL real red-team placement-reweighting audit (review 7a)',
            'provenance': 'results/r2/lanl_blindness/scores_e11_*.npz (read-only); no training',
            'definitions': {'strata': 'per-day cross-edge degree quintiles, K=5',
                            'pooled': 'weighted positives, all negatives (includes cross-stratum terms)',
                            'std': 'unweighted mean of occupied within-stratum AUCs (E32 code convention)',
                            'F': 'min occupied within-stratum AUC',
                            'placement': 'w_s ∝ mean_{pos∈s}(deg+1)^alpha (reweighting, not re-planting)'},
            'evalset': {'days': DAYS, 'arms': ARMS, 'seeds': SEEDS},
            'summary': summ,
        }, fh, indent=1, ensure_ascii=False)
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
