"""r3_e29_equivalence.py — E29：等价界（TOST）裁决层（A席M2 清零；纯既有分布复算）。"""
import json, os
import numpy as np

ROOT = '/root/autodl-tmp/results/r3'
OUT = f'{ROOT}/e29_equivalence'
D1, D2 = 0.02, 0.01


def tost(diffs):
    lo, hi = np.percentile(diffs, [5, 95])
    m = float(diffs.mean())
    return {'mean': round(m, 5), 'ci90': [round(float(lo), 5), round(float(hi), 5)],
            'equiv_d02': bool(lo > -D1 and hi < D1), 'equiv_d01': bool(lo > -D2 and hi < D2)}


rep = {'criteria': {}, 'rows': []}


def from_jsonl(path, filter_fn, base_method):
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for r in rows:
        if not filter_fn(r):
            continue
        key = (r['arm'], r['seed'], r.get('conv', 'asimpl'))
        out.setdefault(key, {})[r['method']] = np.asarray(r['boot_uw'], float)
    res = {}
    for key, arms in sorted(out.items()):
        if base_method not in arms:
            continue
        for m, arr in arms.items():
            if m == base_method:
                continue
            d = arr - arms[base_method]
            res[f'{key[0]}/{key[2]}/{m}'] = {'per_seed': None, **tost(d)}
            res[f'{key[0]}/{key[2]}/{m}']['key'] = [key[0], key[1], key[2], m]
    return res


# E17（blind0，asimpl：heckman − L0）
e17 = {}
rows17 = [json.loads(l) for l in open(f'{ROOT}/e17_ras_ablation/results_e17.jsonl')]
by = {}
for r in rows17:
    by.setdefault((r['arm'], r['seed'], r['conv']), {})[r['method']] = np.asarray(r['boot_uw'], float)
for (arm, seed, conv), arms in sorted(by.items()):
    if 'L0' not in arms:
        continue
    for m in ('heckman', 'L1', 'L2', 'L3', 'L0_fixed0.05'):
        if m in arms:
            t = tost(arms[m] - arms['L0'])
            t.update({'arm': arm, 'seed': seed, 'conv': conv, 'method': m, 'base': 'L0'})
            e17.setdefault(f'{arm}_{conv}_{m}', []).append(t)
rep['E17'] = {k: {'per_seed': v, 'equiv_all_seeds_d02': all(x['equiv_d02'] for x in v),
                  'equiv_all_seeds_d01': all(x['equiv_d01'] for x in v)} for k, v in e17.items()}

# E21（full+arm − full_learned）
e21 = {}
rows21 = [json.loads(l) for l in open(f'{ROOT}/e21_full_correction/results_e21.jsonl')]
by21 = {}
for r in rows21:
    by21.setdefault((r['conv'], r['seed']), {})[r['method']] = np.asarray(r['boot_uw'], float)
for (conv, seed), arms in sorted(by21.items()):
    for m in ('full+L0sel', 'full+heckman', 'full+ctrl_pow', 'full+L0fixed_w0.05'):
        if m in arms and 'full_learned' in arms:
            t = tost(arms[m] - arms['full_learned'])
            t.update({'seed': seed, 'conv': conv, 'method': m, 'base': 'full_learned'})
            e21.setdefault(f'{conv}_{m}', []).append(t)
rep['E21'] = {k: {'per_seed': v, 'equiv_all_seeds_d02': all(x['equiv_d02'] for x in v),
                  'equiv_all_seeds_d01': all(x['equiv_d01'] for x in v)} for k, v in e21.items()}

# E23（npz 重算 heckman − L0sel 的配对正例重采样 bootstrap）
from scipy.stats import norm
from scipy.special import log_ndtr


def imr(z):
    return np.exp(norm.logpdf(z) - log_ndtr(-np.asarray(z, dtype=float)))


e23 = {}
for seed in (0, 1, 2):
    z = np.load(f'{ROOT}/e23_second_domain/scores_e23_seed{seed}.npz', allow_pickle=True)
    dfs, dfn = np.log1p(z['deg_pos'].astype(float)), np.log1p(z['deg_neg'].astype(float))
    hk_p = z['dec_b_pos'] + float(z['rho']) * imr(float(z['gamma']) * dfs)
    hk_n = z['dec_b_neg'] + float(z['rho']) * imr(float(z['gamma']) * dfn)
    l0_p = z['dec_b_pos'] - float(z['w_sel']) * dfs
    l0_n = z['dec_b_neg'] - float(z['w_sel']) * dfn

    def contribs(pos, neg):
        negs = np.sort(neg)
        lf = np.searchsorted(negs, pos, 'left')
        rt = np.searchsorted(negs, pos, 'right')
        return (lf + 0.5 * (rt - lf)) / len(negs)

    c_hk, c_l0 = contribs(hk_p, hk_n), contribs(l0_p, l0_n)
    rng = np.random.RandomState(7)
    n = len(c_hk)
    diffs = np.empty(1000)
    for b in range(1000):
        sel = rng.choice(n, n, replace=True)
        diffs[b] = (c_hk[sel] - c_l0[sel]).mean()
    t = tost(diffs)
    t.update({'seed': seed, 'method': 'heckman', 'base': 'L0sel',
              'auc_heckman': round(float(c_hk.mean()), 4), 'auc_L0sel': round(float(c_l0.mean()), 4)})
    e23.setdefault('enron10_LP_heckman_minus_L0sel', []).append(t)
rep['E23'] = {k: {'per_seed': v, 'equiv_all_seeds_d02': all(x['equiv_d02'] for x in v),
                  'equiv_all_seeds_d01': all(x['equiv_d01'] for x in v)} for k, v in e23.items()}

# 裁决
c = {}
k17 = rep['E17'].get('blind0_asimpl_heckman', {})
c['EQ1'] = {'target': 'E17 blind0/asimpl heckman−L0', 'per_seed': k17.get('per_seed'),
            'equiv_all_seeds_d02': k17.get('equiv_all_seeds_d02'),
            'verdict': ('「统计不可分」成立（90% CI ⊂ ±0.02）' if k17.get('equiv_all_seeds_d02')
                        else '未确立等价 → 措辞改「未检出（报 CI 上限）」')}
k23 = rep['E23'].get('enron10_LP_heckman_minus_L0sel', {})
c['EQ2'] = {'target': 'E23 enron10 LP heckman−L0sel', 'per_seed': k23.get('per_seed'),
            'equiv_all_seeds_d02': k23.get('equiv_all_seeds_d02'),
            'verdict': ('「无特异增益」等价成立' if k23.get('equiv_all_seeds_d02')
                        else '未确立等价 → 措辞改「未检出（报 CI）」')}
k21 = rep['E21'].get('asimpl_full+L0sel', {})
c['EQ3_report'] = {'target': 'E21 asimpl full+L0sel−full_learned',
                   'per_seed': k21.get('per_seed'),
                   'equiv_all_seeds_d02': k21.get('equiv_all_seeds_d02'),
                   'reading': '点估计 +0.0185 ≥ MDE 0.0106 但 90% CI 跨 0 且超 ±0.02 → 「无增量」措辞必须带检测界限定'}
rep['criteria'] = c
json.dump(rep, open(f'{OUT}/analysis_e29.json', 'w'), indent=1, ensure_ascii=False)
open(f'{OUT}/DONE.sentinel', 'w').close()
print(json.dumps(c, ensure_ascii=False, indent=1)[:2000])
print('DONE E29')
