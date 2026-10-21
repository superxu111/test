"""r3_e25_attack_bed.py — E25：受控攻击者床（W5/Q22 + 博弈论 best-response 臂）。

【实验设计（先冻结后跑数；判据 results/r3/e25_attack_bed/criteria_frozen.json）】
验证假设：检测语义盲区「学习分数池化 AUC≈0.51」可由攻击者避枢纽策略解释；度门控评分
（验证日选 w 的 rank(a_learn)−w·log1p(deg_vis)）在避枢纽攻击者下显著优于纯学习评分。
自变量：(a) 攻击者策略 A0_uniform/A1_avoid_b1/A2_avoid_b2/A3_br_learn/A4_br_gate
        （A3/A4=白盒 best-response：候选池成本受限采样 5000 中选评分最低 m 条）；
        (b) 防御者评分器 learn/degfall/gate/pool；(c) 骨干 seed 0-2；(d) 双口径 asimpl/strict。
因变量：日块化池化检测 AUC（跨界层）、植入边可见端度分位中位数、验证日选参数 (w,σ,λ)、低度桶学习 AUC。
对照/baseline：learn 为 baseline；pool=E11 部署法；degfall=纯先验。
规模：LANL 真实拓扑 N=17501，11 评估日 × m=50 植入边 × 5 draw × 3 seed；候选池 20000/日。
成功判据（冻结）：C1 签名剂量 Spearman(β,度分位中位)≤−0.9；C2 learn 崩塌 A0−A2≥0.05 CI 下界>0 3/3；
                C3 方法主张 gate−learn≥0.05 于 A1/A2 CI 下界>0 3/3；C4 BR 稳健性报告型；C5 σ 符号报告型。
防泄漏：植入边仅在评估/验证日边集，不进模型输入；骨干对植入无感知；负例剔除红队边。
防白做：SMOKE=1 小样本（1 seed/2 日/500 候选/30 epochs）跑通全流程后再全量；
        断言 NaN/形状/量级（AUC∈[0,1]、n_pos=11×50、候选度非负）。
产物：results/r3/e25_attack_bed/{results_e25.jsonl, analysis_e25.json, scores_e25_seed*.npz,
      ranks_e25_*.npz, DONE.sentinel}
运行：conda activate euler && python scripts/r3_e25_attack_bed.py 2>&1 | tee logs/r3_e25.log
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '/root/autodl-tmp/scripts')
import r2_e11_lanl_blind as e11  # noqa: E402

OUT = '/root/autodl-tmp/results/r3/e25_attack_bed'
os.makedirs(OUT, exist_ok=True)
SMOKE = bool(os.environ.get('E25_SMOKE'))
EVAL = sorted(e11.EVAL_DAYS)
VAL = [d for d in e11.VAL_RT_DAYS if d <= 9]
TRAIN_DAYS = list(range(0, 10))
CONVS = ('asimpl', 'strict')
W_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
SIG_GRID = (1, -1)
LAM_GRID = (0.3, 0.5, 0.7, 0.9, 1.0)
STRATS = ['S_a1.5', 'S_a1.0', 'S_a0.75', 'S_a0.5', 'S_a0.25', 'A0_uniform',
          'S_am0.5', 'A1_avoid_b1', 'A2_avoid_b2', 'A3_br_learn', 'A4_br_gate']
ALPHA = {'S_a1.5': 1.5, 'S_a1.0': 1.0, 'S_a0.75': 0.75, 'S_a0.5': 0.5, 'S_a0.25': 0.25,
         'A0_uniform': 0.0, 'S_am0.5': -0.5, 'A1_avoid_b1': -1.0, 'A2_avoid_b2': -2.0}
M_PLANT = 50
N_CAND = 500 if SMOKE else 20000
BR_POOL = 200 if SMOKE else 5000
N_BOOT = 500 if SMOKE else 1000
W_BR = 0.05
SEEDS = [0] if SMOKE else [0, 1, 2]
DRAWS = list(range(1 if SMOKE else 5))


def det_seed(*parts):
    return zlib.crc32(':'.join(str(p) for p in parts).encode()) % (2 ** 31)


def ranknorm(v):
    v = np.asarray(v, dtype=float)
    return v.argsort().argsort().astype(float) / max(len(v) - 1, 1)


def joint_rank(x, ns):
    """E17 同构：按段（日）各自 ranknorm 后拼接。"""
    out, ofs = [], 0
    for n in ns:
        out.append(ranknorm(x[ofs:ofs + n]))
        ofs += n
    return np.concatenate(out)


def pct_in(x, ref_sorted):
    lo = np.searchsorted(ref_sorted, x, side='left')
    hi = np.searchsorted(ref_sorted, x, side='right')
    return (lo + 0.5 * (hi - lo)) / max(len(ref_sorted), 1)


def auc_pooled_weighted(pos_ranks, neg_ranks, c):
    """日块加权池化 AUC：pos_ranks/neg_ranks 为 {day: 秩数组}，c 为日权重。
    AUC = Σ_{d1,d2} c1 c2 Cnt(d1,d2) / (Σw_pos Σw_neg)，Cnt=#[r_i>r_j]+0.5#[r_i==r_j]。"""
    days = list(pos_ranks.keys())
    wp = sum(c[d] * len(pos_ranks[d]) for d in days)
    wn = sum(c[d] * len(neg_ranks[d]) for d in days)
    if wp == 0 or wn == 0:
        return None
    num = 0.0
    for d1 in days:
        p1 = pos_ranks[d1]
        if len(p1) == 0 or c[d1] == 0:
            continue
        for d2 in days:
            if c[d2] == 0 or len(neg_ranks[d2]) == 0:
                continue
            nr = neg_ranks[d2]
            lo = np.searchsorted(nr, p1, side='left')
            hi = np.searchsorted(nr, p1, side='right')
            cnt = lo + 0.5 * (hi - lo)
            num += c[d1] * c[d2] * float(cnt.sum())
    return num / (wp * wn)


def strat_indices(strat, n, deg, rng, a=None, conv='asimpl'):
    m = min(M_PLANT, n)
    if strat in ALPHA:
        al = ALPHA[strat]
        if al == 0.0:
            return rng.choice(n, m, replace=False)
        w = (deg + 1.0) ** al
        return rng.choice(n, m, replace=False, p=w / w.sum())
    if strat in ('A3_br_learn', 'A4_br_gate'):
        sub = rng.choice(n, min(BR_POOL, n), replace=False)
        if strat == 'A3_br_learn':
            obj = a[conv][sub]
        else:
            obj = ranknorm(a[conv][sub]) - W_BR * np.log1p(deg[sub])
        return sub[np.argsort(obj)[:m]]
    raise ValueError(strat)


def main():
    crit = json.load(open(f'{OUT}/criteria_frozen.json'))
    print(f'== E25 attack bed | frozen(mtime 为准): {crit["frozen_at"]} | SMOKE={SMOKE}', flush=True)
    ap = argparse.Namespace(epochs=30 if SMOKE else 300, patience=10 if SMOKE else 40, lr=0.005,
                            hidden=32, zdim=16, loss_sample=2000 if SMOKE else 20000)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(8)
    t0 = time.time()
    day_pairs, ev_by_day_pair, N, hosts = e11.load_all()
    S, mask_meta = e11.build_mask(day_pairs, N, OUT)
    Smask = np.zeros(N, dtype=bool)
    Smask[np.asarray(S)] = True
    Vnodes = np.where(~Smask)[0]
    S_arr = np.asarray(S)
    print(f'N={N} |S|={len(S)} ({len(S)/N:.3f})', flush=True)

    def masked_keys(day):
        keys = day_pairs.get(day, np.array([], dtype=np.int64))
        u, v = keys // N, keys % N
        keep = ~(Smask[u] | Smask[v])
        return u[keep], v[keep]

    train_input = [masked_keys(d) for d in TRAIN_DAYS]
    need_days = sorted(set(EVAL) | set(VAL))
    deg_snap = {}
    deg_all = np.zeros(N, dtype=np.int64)
    for d in range(0, max(need_days) + 1):
        if d in need_days:
            deg_snap[d] = deg_all.copy()
        u, v = masked_keys(d)
        np.add.at(deg_all, u, 1)
        np.add.at(deg_all, v, 1)
    rt_by_day = {d: ev_by_day_pair.get(d, {}) for d in need_days}

    all_rows = []
    for seed in SEEDS:
        torch.manual_seed(2026)
        np.random.seed(2026)
        ts = time.time()
        model, best_val, ep_run = e11.train_arm(train_input, N, dev, ap, seed)
        print(f'[seed {seed}] trained best_val={best_val:.4f} epochs={ep_run} ({time.time()-ts:.0f}s)',
              flush=True)
        x = torch.eye(N, device=dev)
        ei_tr = [torch.tensor(np.vstack([u, v]), dtype=torch.long, device=dev)
                 for u, v in train_input]
        with torch.no_grad():
            zs_tr = model(x, ei_tr, lambda i: ei_tr[i])

        day_data = {}
        for d in need_days:
            keys = day_pairs.get(d, np.array([], dtype=np.int64))
            ru, rv = keys // N, keys % N
            m = Smask[ru] ^ Smask[rv]
            ru, rv = ru[m], rv[m]
            mu, mv = masked_keys(d)
            masked_ei = torch.tensor(np.vstack([mu, mv]), dtype=torch.long, device=dev)
            rng_c = np.random.RandomState(det_seed('cand', seed, d))
            cs = S_arr[rng_c.randint(0, len(S_arr), N_CAND)]
            cv = Vnodes[rng_c.randint(0, len(Vnodes), N_CAND)]
            au = np.concatenate([ru, cs]).astype(np.int64)
            av = np.concatenate([rv, cv]).astype(np.int64)
            bu = torch.tensor(au, dtype=torch.long, device=dev)
            bv = torch.tensor(av, dtype=torch.long, device=dev)
            with torch.no_grad():
                if d in VAL and d in TRAIN_DAYS:
                    sc_a = -model.decode(bu, bv, zs_tr[d]).cpu().numpy().astype(np.float64)
                    sc_s = -model.decode(bu, bv, zs_tr[d - 1]).cpu().numpy().astype(np.float64)
                else:
                    seq = ei_tr + [masked_ei]
                    zs = model(x, seq, lambda i: seq[i])
                    sc_a = -model.decode(bu, bv, zs[-1]).cpu().numpy().astype(np.float64)
                    sc_s = -model.decode(bu, bv, zs[len(ei_tr) - 1]).cpu().numpy().astype(np.float64)
            vis_real = np.where(Smask[ru], rv, ru)
            vis_cand = np.where(Smask[cs], cv, cs)
            deg_real = deg_snap[d][vis_real].astype(np.float64)
            deg_cand = deg_snap[d][vis_cand].astype(np.float64)
            assert (deg_real >= 0).all() and (deg_cand >= 0).all()
            ev = np.array([rt_by_day[d].get((min(int(a), int(b)), max(int(a), int(b))), 0)
                           for a, b in zip(ru, rv)], dtype=np.int64)
            day_data[d] = {'ru': ru.astype(np.int32), 'rv': rv.astype(np.int32), 'ev': ev,
                           'deg_real': deg_real,
                           'a_real': {'asimpl': sc_a[:len(ru)], 'strict': sc_s[:len(ru)]},
                           'cs': cs.astype(np.int32), 'cv': cv.astype(np.int32),
                           'deg_cand': deg_cand,
                           'a_cand': {'asimpl': sc_a[len(ru):], 'strict': sc_s[len(ru):]}}
            print(f'  [seed {seed} day {d}] cross={len(ru)} real_ev={int((ev>0).sum())} '
                  f'cand={len(cs)} ({time.time()-ts:.0f}s)', flush=True)

        planted = {}
        for strat in STRATS:
            for dr in DRAWS:
                per_day = {}
                for d in need_days:
                    dd = day_data[d]
                    rng = np.random.RandomState(det_seed(strat, dr, d, seed))
                    per_day[d] = strat_indices(strat, len(dd['cs']), dd['deg_cand'], rng,
                                               a=dd['a_cand'], conv='asimpl')
                planted[(strat, dr)] = per_day

        def make_parts(strat, dr, days):
            parts = []
            for d in days:
                dd = day_data[d]
                idx = planted[(strat, dr)][d]
                keep = dd['ev'] == 0
                parts.append({'day': d,
                              'deg_pos': dd['deg_cand'][idx], 'deg_neg': dd['deg_real'][keep],
                              'a_pos': {c: dd['a_cand'][c][idx] for c in CONVS},
                              'a_neg': {c: dd['a_real'][c][keep] for c in CONVS},
                              'y': np.concatenate([np.ones(len(idx)),
                                                   np.zeros(int(keep.sum()))])})
            return parts

        def select_params(strat, dr, conv):
            """E17 fit_w_levels 同构：gate 的 w 经「原始空间剪切 a−w·log1p(deg) 后逐日 joint-rank 再 AP」选。"""
            P = make_parts(strat, dr, VAL)
            ns = [len(p['y']) for p in P]
            va = np.concatenate([np.concatenate([p['a_pos'][conv], p['a_neg'][conv]]) for p in P])
            vd = np.log1p(np.concatenate([np.concatenate([p['deg_pos'], p['deg_neg']]) for p in P]))
            vlab = np.concatenate([p['y'] for p in P]).astype(int)
            out = {}
            best, bw = -1, 0.0
            for w in W_GRID:
                sc = joint_rank(va - w * vd, ns)
                apv = average_precision_score(vlab, sc)
                if apv > best:
                    best, bw = apv, w
            out['gate_w'] = bw
            out['val_ap_gate'] = round(float(best), 4)
            # 预算缓存：ra/rd 每验证日只算一次（原实现在 λ/σ 循环内重算=245 次排序/单元）
            cache = []
            for p in P:
                a = np.concatenate([p['a_pos'][conv], p['a_neg'][conv]])
                dg = np.log1p(np.concatenate([p['deg_pos'], p['deg_neg']]))
                cache.append((ranknorm(a), ranknorm(dg), p['y']))
            best2, bpair = -1, (0.5, 1)
            for sig in SIG_GRID:
                for lam in LAM_GRID:
                    sc = [lam * ((rd if sig == 1 else 1 - rd)) + (1 - lam) * ra
                          for ra, rd, _ in cache]
                    yy = [p['y'] for p in P]
                    apv = average_precision_score(np.concatenate(yy), np.concatenate(sc))
                    if apv > best2:
                        best2, bpair = apv, (lam, sig)
            out['pool_lam'], out['pool_sig'] = float(bpair[0]), int(bpair[1])
            out['val_ap_pool'] = round(float(best2), 4)
            return out

        def scorer(p, name, params, conv):
            """返回原始分数（AUC 前做逐日 ranknorm=E17 eval_arms 同构）。"""
            a = np.concatenate([p['a_pos'][conv], p['a_neg'][conv]])
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
            raise ValueError(name)

        for strat in STRATS:
            for dr in DRAWS:
                for conv in CONVS:
                    params = select_params(strat, dr, conv)
                    parts = make_parts(strat, dr, EVAL)
                    yy, day_arr = [], []
                    ranks = {'learn': {}, 'degfall': {}, 'gate': {}, 'pool': {}}
                    for p in parts:
                        yy.append(p['y'])
                        day_arr.append(np.full(len(p['y']), p['day'], dtype=np.int32))
                        for name in ranks:
                            ranks[name][p['day']] = ranknorm(scorer(p, name, params, conv))
                    yy = np.concatenate(yy)
                    day_arr = np.concatenate(day_arr)
                    y_by_day = {p['day']: p['y'] for p in parts}
                    pos_ranks = {n: {d: ranks[n][d][y_by_day[d] == 1] for d in EVAL} for n in ranks}
                    neg_ranks = {n: {d: np.sort(ranks[n][d][y_by_day[d] == 0]) for d in EVAL}
                                 for n in ranks}
                    c1 = {d: 1 for d in EVAL}
                    aucs = {n: round(float(auc_pooled_weighted(pos_ranks[n], neg_ranks[n], c1)), 4)
                            for n in ranks}
                    assert all(0.0 <= v <= 1.0 for v in aucs.values()), aucs
                    pcts = [pct_in(p['deg_pos'], np.sort(p['deg_neg'])) for p in parts]
                    med_pct = float(np.median(np.concatenate(pcts)))
                    bt = []
                    for p in parts:
                        dg = np.concatenate([p['deg_pos'], p['deg_neg']])
                        thr = np.quantile(dg, 0.5)
                        sel = dg <= thr
                        if p['y'][sel].sum() > 0 and (1 - p['y'][sel]).sum() > 0:
                            bt.append(float(roc_auc_score(p['y'][sel], ranks['learn'][p['day']][sel])))
                    assert int((yy == 1).sum()) == len(EVAL) * M_PLANT, int((yy == 1).sum())
                    row = {'seed': seed, 'strategy': strat, 'draw': dr, 'conv': conv,
                           'auc': aucs, 'params': params, 'median_deg_pct': round(med_pct, 4),
                           'lowdeg_learn_auc': round(float(np.mean(bt)), 4) if bt else None,
                           'n_pos': int((yy == 1).sum()), 'n_neg': int((yy == 0).sum())}
                    all_rows.append(row)
                    print(f'[seed {seed} {strat} d{dr} {conv}] {json.dumps(aucs)} medpct='
                          f'{med_pct:.3f} w={params["gate_w"]} pool=({params["pool_lam"]},'
                          f'{params["pool_sig"]})', flush=True)
                    np.savez_compressed(
                        f'{OUT}/ranks_e25_{seed}_{strat}_d{dr}_{conv}.npz', y=yy, day=day_arr,
                        **{f'r_{k}': np.concatenate([ranks[k][d] for d in EVAL]) for k in ranks})

        npz = {}
        for d in need_days:
            dd = day_data[d]
            npz[f'{d}_cross_u'] = dd['ru']
            npz[f'{d}_cross_v'] = dd['rv']
            npz[f'{d}_cross_ev'] = dd['ev']
            npz[f'{d}_cross_deg'] = dd['deg_real'].astype(np.float32)
            for c in CONVS:
                npz[f'{d}_cross_a_{c}'] = dd['a_real'][c].astype(np.float32)
            npz[f'{d}_cand_s'] = dd['cs']
            npz[f'{d}_cand_v'] = dd['cv']
            npz[f'{d}_cand_deg'] = dd['deg_cand'].astype(np.float32)
            for c in CONVS:
                npz[f'{d}_cand_a_{c}'] = dd['a_cand'][c].astype(np.float32)
        np.savez_compressed(f'{OUT}/scores_e25_seed{seed}.npz', **npz)
        json.dump({'mask': mask_meta}, open(f'{OUT}/mask_meta.json', 'w'))

    with open(f'{OUT}/results_e25.jsonl', 'w') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')

    from scipy import stats as sps

    def boot(seed, s1, s2, n1, n2, conv='asimpl'):
        z1 = [np.load(f'{OUT}/ranks_e25_{seed}_{s1}_d{dr}_{conv}.npz') for dr in DRAWS]
        z2 = [np.load(f'{OUT}/ranks_e25_{seed}_{s2}_d{dr}_{conv}.npz') for dr in DRAWS]
        pr1 = {d: np.concatenate([z[f'r_{n1}'][z['day'] == d][z['y'][z['day'] == d] == 1]
                                  for z in z1]) for d in EVAL}
        nr1 = {d: np.sort(np.concatenate([z[f'r_{n1}'][z['day'] == d][z['y'][z['day'] == d] == 0]
                                          for z in z1])) for d in EVAL}
        pr2 = {d: np.concatenate([z[f'r_{n2}'][z['day'] == d][z['y'][z['day'] == d] == 1]
                                  for z in z2]) for d in EVAL}
        nr2 = {d: np.sort(np.concatenate([z[f'r_{n2}'][z['day'] == d][z['y'][z['day'] == d] == 0]
                                          for z in z2])) for d in EVAL}
        rng = np.random.RandomState(7)
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            sel = rng.choice(len(EVAL), len(EVAL), replace=True)
            c = {d: int((sel == i).sum()) for i, d in enumerate(EVAL)}
            diffs[b] = (auc_pooled_weighted(pr1, nr1, c) - auc_pooled_weighted(pr2, nr2, c))
        return {'mean': round(float(diffs.mean()), 4),
                'ci95': [round(float(np.percentile(diffs, 2.5)), 4),
                         round(float(np.percentile(diffs, 97.5)), 4)],
                'frac_gt0': round(float((diffs > 0).mean()), 4)}

    analysis = {'criteria': {}, 'per_seed': {}, 'interpretation_limits': crit['interpretation_limits']}
    for seed in SEEDS:
        det = {}
        for strat in STRATS:
            for r in all_rows:
                if r['seed'] == seed and r['strategy'] == strat and r['conv'] == 'asimpl':
                    det.setdefault(strat, []).append(
                        {'draw': r['draw'], 'auc': r['auc'], 'median_deg_pct': r['median_deg_pct'],
                         'gate_w': r['params']['gate_w'], 'pool_sig': r['params']['pool_sig'],
                         'lowdeg_learn_auc': r['lowdeg_learn_auc']})
        analysis['per_seed'][str(seed)] = det

    alpha_pts = [(ALPHA[s], s) for s in STRATS if s in ALPHA]
    c1 = {'alpha_to_medpct': {}}
    xs, ys_ = [], []
    for al, s in sorted(alpha_pts):
        vals = [r['median_deg_pct'] for r in all_rows
                if r['strategy'] == s and r['conv'] == 'asimpl']
        c1['alpha_to_medpct'][f'{al:+.2f}'] = round(float(np.mean(vals)), 4)
        xs.append(al)
        ys_.append(float(np.mean(vals)))
    rho = sps.spearmanr(xs, ys_)
    c1['spearman'] = round(float(rho.statistic), 4)
    c1['pass'] = bool(rho.statistic <= -0.9)
    analysis['criteria']['C1_signature_dose'] = c1

    def verdict(key, s1, s2, n1, n2, sign=1, hard=True):
        rows = {str(s): boot(s, s1, s2, n1, n2) for s in SEEDS}
        means = [sign * rows[str(s)]['mean'] for s in SEEDS]
        lbs = [rows[str(s)]['ci95'][0] if sign == 1 else -rows[str(s)]['ci95'][1] for s in SEEDS]
        out = {'per_seed': rows, 'mean_of_means': round(float(np.mean(means)), 4),
               'all_means_ge_0.05': bool(all(m >= 0.05 for m in means)),
               'all_ci_lower_gt0': bool(all(l > 0 for l in lbs))}
        out['pass'] = bool(out['all_means_ge_0.05'] and out['all_ci_lower_gt0']) if hard else None
        out['supports_claim'] = bool(out['all_means_ge_0.05'] and out['all_ci_lower_gt0'])
        analysis['criteria'][key] = out

    verdict('C2_learn_collapse', 'A0_uniform', 'A2_avoid_b2', 'learn', 'learn')
    verdict('C3_gate_minus_learn_A1', 'A1_avoid_b1', 'A1_avoid_b1', 'gate', 'learn')
    verdict('C3_gate_minus_learn_A2', 'A2_avoid_b2', 'A2_avoid_b2', 'gate', 'learn')
    verdict('C4_gate_minus_learn_A3', 'A3_br_learn', 'A3_br_learn', 'gate', 'learn', hard=False)
    verdict('C4_gate_minus_learn_A4', 'A4_br_gate', 'A4_br_gate', 'gate', 'learn', hard=False)
    verdict('C4_pool_minus_learn_A2', 'A2_avoid_b2', 'A2_avoid_b2', 'pool', 'learn', hard=False)

    # C6：α* 标定（|learn−0.5019| 最小）
    real_learn, real_pct, real_gate = 0.5019, 0.4854, 0.9432
    tab = {}
    for al, s in alpha_pts:
        rows = [r for r in all_rows if r['strategy'] == s and r['conv'] == 'asimpl']
        tab[s] = {'alpha': al,
                  'learn': round(float(np.mean([r['auc']['learn'] for r in rows])), 4),
                  'degfall': round(float(np.mean([r['auc']['degfall'] for r in rows])), 4),
                  'gate': round(float(np.mean([r['auc']['gate'] for r in rows])), 4),
                  'pool': round(float(np.mean([r['auc']['pool'] for r in rows])), 4),
                  'medpct': round(float(np.mean([r['median_deg_pct'] for r in rows])), 4),
                  'gate_w_mode': float(sps.mode([r['params']['gate_w'] for r in rows],
                                                keepdims=False).mode),
                  'lowdeg_learn': round(float(np.mean([r['lowdeg_learn_auc'] for r in rows])), 4)}
    a_star = min(tab.values(), key=lambda v: abs(v['learn'] - real_learn))
    c6 = {'table': tab, 'real_targets': {'learn': real_learn, 'medpct': real_pct,
                                         'gate_sweetspot': real_gate},
          'alpha_star': a_star['alpha'], 'at_alpha_star': a_star,
          'learn_match': round(abs(a_star['learn'] - real_learn), 4),
          'pct_match': round(abs(a_star['medpct'] - real_pct), 4),
          'gate_reproduces_real_sweetspot': bool(a_star['gate'] >= 0.85)}
    c6['pass'] = bool(c6['learn_match'] <= 0.02 and c6['pct_match'] <= 0.15)
    analysis['criteria']['C6_alpha_calibration'] = c6

    # C7：隐身攻击者（α=+1.0）
    s_blend = 'S_a1.0'
    rows = [r for r in all_rows if r['strategy'] == s_blend and r['conv'] == 'asimpl']
    c7 = {'alpha': 1.0,
          'degfall': round(float(np.mean([r['auc']['degfall'] for r in rows])), 4),
          'learn': round(float(np.mean([r['auc']['learn'] for r in rows])), 4),
          'gate': round(float(np.mean([r['auc']['gate'] for r in rows])), 4),
          'gate_w_mode': float(sps.mode([r['params']['gate_w'] for r in rows], keepdims=False).mode),
          'prediction': 'degfall≈0.5（度信号消失）时 gate 应自动退化（w 选择小/近 learn）'}
    c7['degfall_signal_vanishes'] = bool(abs(c7['degfall'] - 0.5) <= 0.05)
    c7['gate_degrades_to_learn'] = bool(abs(c7['gate'] - c7['learn']) <= 0.05)
    c7['pass'] = None
    analysis['criteria']['C7_blendin'] = c7

    c5 = {}
    for strat in STRATS:
        sigs = [r['params']['pool_sig'] for r in all_rows
                if r['strategy'] == strat and r['conv'] == 'asimpl']
        c5[strat] = {'sigs': sigs, 'mode': int(sps.mode(sigs, keepdims=False).mode)}
    c5['prediction'] = 'σ*(A1/A2)=−1（避枢纽）；真实数据 E11 LANL σ*=−1 同向（报告型）'
    c5['pass'] = bool(c5['A1_avoid_b1']['mode'] == -1 and c5['A2_avoid_b2']['mode'] == -1)
    analysis['criteria']['C5_sigma_sign'] = c5

    analysis['n_rows'] = len(all_rows)
    analysis['smoke'] = SMOKE
    analysis['elapsed_sec'] = round(time.time() - t0, 1)
    json.dump(analysis, open(f'{OUT}/analysis_e25.json', 'w'), indent=1, ensure_ascii=False)
    open(f'{OUT}/DONE.sentinel', 'w').close()
    print('criteria:', json.dumps({k: v.get('pass') for k, v in analysis['criteria'].items()},
                                  ensure_ascii=False), flush=True)
    print(f'DONE E25 ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
