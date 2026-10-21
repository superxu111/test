"""r3_e31_lib.py — E31/E32/E33 共用底层：从既有 E25 工件零重训重建/复算。

设计要点：
  * 重建（planting 复生、候选池、真实边分）与 scripts/r3_e25_attack_bed.py 逐位同构；
  * 度量用「日对计数矩阵」表示：AUC_c = c^T Cnt c / (c^T npos)(c^T nneg)，任意日权重/分层都 O(121)；
    符号翻转（rank→1-rank）精确给出 Cnt_flip = npos*nneg - Cnt（连续秩下 #> + #< + #= 恒等）；
  * 分层（K 层，可见端度等频）逐层同构计数 → 池化 / 标准化（顶层均权）/ 最坏层下界 F 三口径。
"""
import json, os, sys, zlib
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/scripts')
BED = '/root/autodl-tmp/results/r3/e25_attack_bed'
EVAL = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
VAL = [1, 2, 5, 6, 7, 8, 9]
ALLD = VAL + EVAL
CONVS = ('asimpl', 'strict')
W_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SIG_GRID = (1, -1)
LAM_GRID = (0.3, 0.5, 0.7, 0.9, 1.0)
STRATS = ['S_a1.5', 'S_a1.0', 'S_a0.75', 'S_a0.5', 'S_a0.25', 'A0_uniform',
          'S_am0.5', 'A1_avoid_b1', 'A2_avoid_b2', 'A3_br_learn', 'A4_br_gate']
ALPHA = {'S_a1.5': 1.5, 'S_a1.0': 1.0, 'S_a0.75': 0.75, 'S_a0.5': 0.5, 'S_a0.25': 0.25,
         'A0_uniform': 0.0, 'S_am0.5': -0.5, 'A1_avoid_b1': -1.0, 'A2_avoid_b2': -2.0}
M_PLANT, N_CAND, BR_POOL, W_BR = 50, 20000, 5000, 0.05


def det_seed(*parts):
    return zlib.crc32(':'.join(str(p) for p in parts).encode()) % (2 ** 31)


def ranknorm(v):
    v = np.asarray(v, dtype=float)
    return v.argsort().argsort().astype(float) / max(len(v) - 1, 1)


def pct_in(x, ref_sorted):
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return (lo + 0.5 * (hi - lo)) / max(len(ref_sorted), 1)


def joint_rank(x, ns):
    out, ofs = [], 0
    for n in ns:
        out.append(ranknorm(x[ofs:ofs + n]))
        ofs += n
    return np.concatenate(out)


def strat_indices(strat, n, deg, rng, a=None, conv='asimpl', w_att=None):
    """E25 逐位同构；w_att 非 None → A4′ 指定攻击目标权重（免训臂）。"""
    m = min(M_PLANT, n)
    if strat in ALPHA:
        al = ALPHA[strat]
        if al == 0.0:
            return rng.choice(n, m, replace=False)
        w = (deg + 1.0) ** al
        return rng.choice(n, m, replace=False, p=w / w.sum())
    sub = rng.choice(n, min(BR_POOL, n), replace=False)
    if strat == 'A3_br_learn':
        obj = a[conv][sub]
    else:
        ww = W_BR if w_att is None else w_att
        obj = ranknorm(a[conv][sub]) - ww * np.log1p(deg[sub])
    return sub[np.argsort(obj)[:m]]


_SCO = {}


def scores(seed):
    if seed not in _SCO:
        _SCO[seed] = np.load(f'{BED}/scores_e25_seed{seed}.npz', allow_pickle=True)
    return _SCO[seed]


def cand_index(seed, strat, dr, d, w_att=None):
    """复生某日植入索引（E25 逐位同构：asimpl 口径候选池 a 分）。"""
    zs = scores(seed)
    deg_cand = zs[f'{d}_cand_deg'].astype(np.float64)
    a_cand = zs[f'{d}_cand_a_asimpl'].astype(np.float64)
    rng = np.random.RandomState(det_seed(strat, dr, d, seed))
    return strat_indices(strat, len(deg_cand), deg_cand, rng, a={'asimpl': a_cand},
                         conv='asimpl', w_att=w_att)


def load_case(seed, strat, dr, conv):
    """E25 逐位同构的 EVAL 逐日数组：y/deg/r_*(npz 秩)。"""
    z = np.load(f'{BED}/ranks_e25_{seed}_{strat}_d{dr}_{conv}.npz')
    y_all, day_all = z['y'], z['day']
    rk = {n: z[f'r_{n}'] for n in ('learn', 'degfall', 'gate', 'pool')}
    zs = scores(seed)
    out = {}
    ofs = 0
    for d in EVAL:
        nd = int((day_all == d).sum())
        y = y_all[ofs:ofs + nd]
        idx = cand_index(seed, strat, dr, d)
        keep = zs[f'{d}_cross_ev'] == 0
        rec = {'y': y, 'deg': np.concatenate([zs[f'{d}_cand_deg'][idx].astype(np.float64),
                                              zs[f'{d}_cross_deg'][keep].astype(np.float64)]),
               'n_pos': len(idx), 'n_neg': int(keep.sum())}
        assert len(rec['y']) == nd and int(y.sum()) == len(idx)
        for n in rk:
            rec[f'r_{n}'] = rk[n][ofs:ofs + nd]
        out[d] = rec
        ofs += nd
    return out


# ---------------- 原始分（含 VAL 日）：E25 scorer 同构 ----------------

def raw_case(seed, strat, dr, conv, days=None):
    """{day: {y, deg_pos/neg, a_pos/neg(conv 口径原始分), idx}}；正例=植入，负例=当日跨界非红队边。"""
    days = ALLD if days is None else days
    zs = scores(seed)
    out = {}
    for d in days:
        idx = cand_index(seed, strat, dr, d)
        keep = zs[f'{d}_cross_ev'] == 0
        out[d] = {'idx': idx,
                  'deg_pos': zs[f'{d}_cand_deg'][idx].astype(np.float64),
                  'deg_neg': zs[f'{d}_cross_deg'][keep].astype(np.float64),
                  'a_pos': zs[f'{d}_cand_a_{conv}'][idx].astype(np.float64),
                  'a_neg': zs[f'{d}_cross_a_{conv}'][keep].astype(np.float64),
                  'y': np.concatenate([np.ones(len(idx)), np.zeros(int(keep.sum()))])}
    return out


def case_extra_br(seed, dr, days, w_att=None, target='gate'):
    """A4′：正例=按目标评分最低的 50 条（候选池），其余同 raw_case。"""
    zs = scores(seed)
    out = {}
    for d in days:
        idx = plant_br(seed, d, zs, w_att=w_att, target=target)
        keep = zs[f'{d}_cross_ev'] == 0
        out[d] = {'idx': idx,
                  'deg_pos': zs[f'{d}_cand_deg'][idx].astype(np.float64),
                  'deg_neg': zs[f'{d}_cross_deg'][keep].astype(np.float64),
                  'a_pos': zs[f'{d}_cand_a_asimpl'][idx].astype(np.float64),
                  'a_neg': zs[f'{d}_cross_a_asimpl'][keep].astype(np.float64),
                  'a_pos_strict': zs[f'{d}_cand_a_strict'][idx].astype(np.float64),
                  'a_neg_strict': zs[f'{d}_cross_a_strict'][keep].astype(np.float64),
                  'y': np.concatenate([np.ones(len(idx)), np.zeros(int(keep.sum()))])}
    return out


def raw_case_cross(seed, arm, dr, days=None):
    """A5 臂：正例 = 当日真实跨界非红队边（度分位逐位匹配 S_a1.5 或均匀）；负例 = 同集 − 正例。"""
    days = ALLD if days is None else days
    zs = scores(seed)
    out = {}
    for d in days:
        deg = zs[f'{d}_cross_deg'].astype(np.float64)
        keep = zs[f'{d}_cross_ev'] == 0
        cand_rows = np.where(keep)[0]
        rng = np.random.RandomState(det_seed(arm, dr, d, seed))
        if arm == 'A5_exist_uniform':
            sel = rng.choice(len(cand_rows), M_PLANT, replace=False)
        elif arm == 'A5_exist_matched':
            tgt = cand_index(seed, 'S_a1.5', dr, d)
            deg_cand_arr = zs[f'{d}_cand_deg'].astype(np.float64)
            ref = np.sort(deg[keep])
            r_tgt = np.searchsorted(ref, deg_cand_arr[tgt], side='left')
            r_cand = np.searchsorted(ref, deg[cand_rows], side='left')
            o = np.argsort(r_cand, kind='mergesort')
            r_sorted = r_cand[o]
            used = np.zeros(len(cand_rows), dtype=bool)
            sel = np.empty(M_PLANT, dtype=np.int64)
            for i, rt in enumerate(r_tgt):
                j = int(np.argmin(np.abs(np.where(used, 10 ** 12, r_sorted) - rt)))
                used[j] = True
                sel[i] = o[j]
        else:
            raise ValueError(arm)
        pos_rows = cand_rows[sel]
        neg_mask = keep.copy()
        neg_mask[pos_rows] = False
        out[d] = {'idx': pos_rows,
                  'deg_pos': deg[pos_rows], 'deg_neg': deg[neg_mask],
                  'a_pos': zs[f'{d}_cross_a_asimpl'][pos_rows].astype(np.float64),
                  'a_neg': zs[f'{d}_cross_a_asimpl'][neg_mask].astype(np.float64),
                  'a_pos_strict': zs[f'{d}_cross_a_strict'][pos_rows].astype(np.float64),
                  'a_neg_strict': zs[f'{d}_cross_a_strict'][neg_mask].astype(np.float64),
                  'y': np.concatenate([np.ones(M_PLANT), np.zeros(int(neg_mask.sum()))])}
    return out


def plant_br(seed, d, zs, w_att=None, target='gate'):
    """A4′ 攻击者：候选池 5000 子采样内按目标评分最低的 50 条（E25 A4 逐位同构 + 目标扩展）。"""
    deg_cand = zs[f'{d}_cand_deg'].astype(np.float64)
    a_cand = zs[f'{d}_cand_a_asimpl'].astype(np.float64)
    rng = np.random.RandomState(det_seed('A4p', target, w_att, d, seed))
    sub = rng.choice(len(deg_cand), min(BR_POOL, len(deg_cand)), replace=False)
    if target == 'learn':
        obj = a_cand[sub]  # E25 A3_br_learn 逐位同构：升序取最低 learn 分（a 越高越可疑）
    elif target == 'cas':
        obj = cas_proxy(a_cand[sub], deg_cand[sub])
    else:
        obj = ranknorm(a_cand[sub]) - float(w_att) * np.log1p(deg_cand[sub])
    return sub[np.argsort(obj)[:M_PLANT]]


def cas_proxy(a, deg, K=5, sig=1):
    """攻击者侧 CAS 代理：候选池内按度分位 K 层的层内秩组合（0.5/0.5 固定约定）。"""
    q = np.clip((ranknorm(deg) * K).astype(int), 0, K - 1)
    zl = np.zeros(len(a)); zd = np.zeros(len(a))
    for s in range(K):
        m = q == s
        if m.sum() == 0:
            continue
        zl[m] = ranknorm(a[m])
        zd[m] = ranknorm((sig * np.log1p(deg))[m])
    return 0.5 * zl + 0.5 * zd


# ---------------- 参数选择与评分（E25 逐位同构） ----------------

def _a_pair(p, conv):
    if conv == 'strict' and 'a_pos_strict' in p:
        return p['a_pos_strict'], p['a_neg_strict']
    return p['a_pos'], p['a_neg']


def select_params_val(case, conv):
    """在验证日 parts 上选 (gate_w, pool_lam, pool_sig)——r3_e25_attack_bed.py select_params 同构。"""
    from sklearn.metrics import average_precision_score
    P = [case[d] for d in VAL]
    ns = [len(p['y']) for p in P]
    va = np.concatenate([np.concatenate(list(_a_pair(p, conv))) for p in P])
    vd = np.log1p(np.concatenate([np.concatenate([p['deg_pos'], p['deg_neg']]) for p in P]))
    vlab = np.concatenate([p['y'] for p in P]).astype(int)
    best, bw = -1, 0.0
    for w in W_GRID:
        apv = average_precision_score(vlab, joint_rank(va - w * vd, ns))
        if apv > best:
            best, bw = apv, w
    cache = []
    for p in P:
        a = np.concatenate(list(_a_pair(p, conv)))
        dg = np.log1p(np.concatenate([p['deg_pos'], p['deg_neg']]))
        cache.append((ranknorm(a), ranknorm(dg), p['y']))
    best2, bpair = -1, (0.5, 1)
    for sig in SIG_GRID:
        for lam in LAM_GRID:
            sc = [lam * ((rd if sig == 1 else 1 - rd)) + (1 - lam) * ra for ra, rd, _ in cache]
            apv = average_precision_score(np.concatenate([c[2] for c in cache]),
                                          np.concatenate(sc))
            if apv > best2:
                best2, bpair = apv, (lam, sig)
    return {'gate_w': bw, 'pool_lam': float(bpair[0]), 'pool_sig': int(bpair[1])}


def scorer_raw(p, name, params, conv):
    """E25 scorer 同构（原始空间）。"""
    a = np.concatenate(list(_a_pair(p, conv)))
    dg = np.log1p(np.concatenate([p['deg_pos'], p['deg_neg']]))
    if name == 'learn':
        return a
    if name == 'degfall':
        return dg if params['pool_sig'] == 1 else -dg
    if name == 'gate':
        return a - params['gate_w'] * dg
    if name == 'pool':
        ra, rd = ranknorm(a), ranknorm(dg)
        lam, sig = params['pool_lam'], params['pool_sig']
        return lam * (rd if sig == 1 else 1 - rd) + (1 - lam) * ra
    if name == 'cas':
        return cas_case_ranks(p, params['pool_sig'])
    raise ValueError(name)


def cas_case_ranks(p, sig, K=5):
    """CAS（冻结定义）：CAS = 0.5·z_learn + 0.5·z_deg；z=层内秩；σ=验证日选出的度方向；无跨层项。"""
    deg = np.concatenate([p['deg_pos'], p['deg_neg']])
    a = np.concatenate([p['a_pos'], p['a_neg']])
    q = np.clip((ranknorm(deg) * K).astype(int), 0, K - 1)
    zl = np.zeros(len(a)); zd = np.zeros(len(a))
    for s in range(K):
        m = q == s
        if m.sum() == 0:
            continue
        zl[m] = ranknorm(a[m])
        zd[m] = ranknorm((sig * np.log1p(deg))[m])
    return 0.5 * zl + 0.5 * zd


def ranks_day(case, day, names, params, conv):
    p = case[day]
    return {n: ranknorm(scorer_raw(p, n, params, conv)) for n in names}


def deg_day(case, day):
    p = case[day]
    return np.concatenate([p['deg_pos'], p['deg_neg']])


# ---------------- 计数矩阵度量 ----------------

def cnt_entry(pos, neg_sorted):
    """pos/neg_sorted 为同一天内的秩数组 → (#>+0.5#=, npos, nneg)。"""
    if len(pos) == 0 or len(neg_sorted) == 0:
        return 0.0, len(pos), len(neg_sorted)
    lo = np.searchsorted(neg_sorted, pos, 'left')
    hi = np.searchsorted(neg_sorted, pos, 'right')
    return float((lo + 0.5 * (hi - lo)).sum()), len(pos), len(neg_sorted)


def pack(ranks_by_day, y_by_day, days, strat_key_map=None, K=None):
    """计数包：{'glob': (Cnt, npos, nneg), 'strata': {s: (...)} }；Cnt[i,j] = pos(day i) vs neg(day j)。"""
    nd = len(days)
    Cg = np.zeros((nd, nd)); pg = np.zeros(nd); ng = np.zeros(nd)
    Cs = {s: np.zeros((nd, nd)) for s in range(K)} if K else None
    ps = {s: np.zeros(nd) for s in range(K)} if K else None
    ns_ = {s: np.zeros(nd) for s in range(K)} if K else None
    negs_sorted = {}
    strat_negs = {}
    for i, d in enumerate(days):
        y = y_by_day[d]; r = ranks_by_day[d]
        negs_sorted[i] = np.sort(r[y == 0])
        pg[i], ng[i] = int((y == 1).sum()), int((y == 0).sum())
        if K:
            q = strat_key_map[d]
            for s in range(K):
                m = q == s
                ps[s][i] = int((m & (y == 1)).sum())
                ns_[s][i] = int((m & (y == 0)).sum())
                strat_negs[(s, i)] = np.sort(r[m & (y == 0)])
    for i, d in enumerate(days):
        y = y_by_day[d]; r = ranks_by_day[d]
        p1 = r[y == 1]
        for j in range(nd):
            c, _, _ = cnt_entry(p1, negs_sorted[j])
            Cg[i, j] = c
            if K:
                q = strat_key_map[d]
                for s in range(K):
                    m = q == s
                    c2, _, _ = cnt_entry(r[m & (y == 1)], strat_negs[(s, j)])
                    Cs[s][i, j] = c2
    out = {'glob': (Cg, pg, ng), 'days': list(days)}
    if K:
        out['strata'] = {s: (Cs[s], ps[s], ns_[s]) for s in range(K)}
    return out


def auc_from(entry, c):
    C, p, n = entry
    wp = float((c * p).sum()); wn = float((c * n).sum())
    if wp <= 0 or wn <= 0:
        return None
    return float(c @ C @ c) / (wp * wn)


def flip_entry(entry):
    C, p, n = entry
    return (p[:, None] * n[None, :] - C, p, n)


def metrics_from_pk(pk, c, names=('learn', 'degfall', 'gate', 'pool', 'cas')):
    out = {}
    for nm in names:
        if nm not in pk:
            continue
        P = pk[nm]
        pooled = auc_from(P['glob'], c)
        per = {s: auc_from(e, c) for s, e in P['strata'].items()}
        vals = [v for v in per.values() if v is not None]
        out[nm] = {'pooled': None if pooled is None else round(pooled, 6),
                   'std': None if not vals else round(float(np.mean(vals)), 6),
                   'F': None if not vals else round(float(np.min(vals)), 6),
                   'per_stratum': {s: (None if v is None else round(v, 6)) for s, v in per.items()}}
    return out


def flip_pk_metrics(pk, c, names):
    """翻转符号（rank→1-rank）后的三口径（精确）。"""
    out = {}
    for nm in names:
        if nm not in pk:
            continue
        P = pk[nm]
        pooled = auc_from(flip_entry(P['glob']), c)
        vals = [auc_from(flip_entry(e), c) for e in P['strata'].values()]
        vals = [v for v in vals if v is not None]
        out[nm] = {'pooled': pooled, 'std': (None if not vals else float(np.mean(vals))),
                   'F': (None if not vals else float(np.min(vals)))}
    return out


def strat_key_of(case, days, K):
    return {d: np.clip((ranknorm(deg_day(case, d)) * K).astype(int), 0, K - 1) for d in days}


def boot_weights(days, n_boot, seed=7):
    rng = np.random.RandomState(seed)
    nd = len(days)
    for _ in range(n_boot):
        sel = rng.choice(nd, nd, replace=True)
        c = np.bincount(sel, minlength=nd).astype(float)
        yield c
