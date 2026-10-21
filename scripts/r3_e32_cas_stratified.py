"""r3_e32_cas_stratified.py — E32：CAS 条件分层评分 + 三口径（池化/标准化/最坏层下界）+ oracle 符号最强平凡基线。

判据：results/r3/e32_cas_stratified/criteria_frozen.json（先冻结后跑数；打印判据 mtime 与产物 mtime）
运行：conda activate euler && setsid nohup python scripts/r3_e32_cas_stratified.py > logs/r3_e32.log 2>&1 &
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/scripts')
import r3_e31_lib as L

OUT = '/root/autodl-tmp/results/r3/e32_cas_stratified'
CRIT = f'{OUT}/criteria_frozen.json'
NAMES = ('learn', 'degfall', 'gate', 'pool', 'cas')
N_BOOT = 1000
K_MAIN = 5
MIN_GAP = 0.0106   # E12 MDE（冻结文本引用）
WIN = ['S_a0.75', 'S_a1.0', 'S_a1.5']
EXTREME = ['A0_uniform', 'S_am0.5', 'A1_avoid_b1', 'A2_avoid_b2', 'A3_br_learn', 'A4_br_gate']
C1 = np.ones(len(L.EVAL))


def cas_ranks(rec, sig, K):
    """CAS（冻结定义，见 criteria_frozen.definitions_frozen.CAS）：层内秩 0.5/0.5，无跨层项。"""
    deg = rec['deg']
    a = rec['r_learn']
    q = np.clip((L.ranknorm(deg) * K).astype(int), 0, K - 1)
    zl = np.zeros(len(a)); zd = np.zeros(len(a))
    for s in range(K):
        m = q == s
        if m.sum() == 0:
            continue
        zl[m] = L.ranknorm(a[m])
        zd[m] = L.ranknorm((sig * np.log1p(deg))[m])
    return 0.5 * zl + 0.5 * zd


def build_entries(seed, strat, dr, conv, params, K):
    """返回 {'cas': (glob_e, {s: e}), 'base': [(glob_e, {s:e}) ×6]}（6=3 scorer × 2 符号）。"""
    case = L.load_case(seed, strat, dr, conv)
    yb = {d: case[d]['y'] for d in L.EVAL}
    sk = {d: np.clip((L.ranknorm(case[d]['deg']) * K).astype(int), 0, K - 1) for d in L.EVAL}

    def pk_of(ranks_by_day):
        return L.pack(ranks_by_day, yb, L.EVAL, sk, K)

    def split(pk):
        return pk['glob'], pk['strata']

    ranks = {nm: {d: case[d][f'r_{nm}'] for d in L.EVAL} for nm in ('learn', 'degfall', 'gate', 'pool')}
    ranks['cas'] = {d: L.ranknorm(cas_ranks(case[d], params['pool_sig'], K)) for d in L.EVAL}
    entries = {'cases': {nm: split(pk_of(ranks[nm])) for nm in NAMES}}
    base = []
    for nm in ('degfall', 'learn', 'pool'):
        g, st = entries['cases'][nm]
        fg = L.flip_entry(g)
        fst = {s: L.flip_entry(e) for s, e in st.items()}
        base.extend([(g, st), (fg, fst)])
    entries['base'] = base
    return entries


def triple(entry, c):
    g, st = entry
    pooled = L.auc_from(g, c)
    vals = [L.auc_from(e, c) for e in st.values()]
    vals = [v for v in vals if v is not None]
    return {'pooled': pooled, 'std': (float(np.mean(vals)) if vals else None),
            'F': (float(np.min(vals)) if vals else None)}


def metrics_all(entries, c):
    out = {}
    for nm, e in entries['cases'].items():
        out[nm] = triple(e, c)
    out['oracle'] = {k: max(v[k] for v in (triple(b, c) for b in entries['base'])
                            if v[k] is not None)
                     for k in ('pooled', 'std', 'F')}
    return out


def main():
    t0 = time.time()
    cm = os.path.getmtime(CRIT)
    print(f'== E32 | criteria mtime={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cm))} '
          f'| now={time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    ref = {(r['seed'], r['strategy'], r['draw'], r['conv']): r
           for r in (json.loads(l) for l in open(f'{L.BED}/results_e25.jsonl'))}

    # ---------- 全表（K=5） ----------
    ENT = {}
    lines = []
    for seed in (0, 1, 2):
        for strat in L.STRATS:
            for dr in range(5):
                for conv in L.CONVS:
                    params = ref[(seed, strat, dr, conv)]['params']
                    e = build_entries(seed, strat, dr, conv, params, K_MAIN)
                    ENT[(seed, strat, dr, conv)] = e
                    m = metrics_all(e, C1)
                    lines.append({'seed': seed, 'strategy': strat, 'draw': dr, 'conv': conv,
                                  'alpha': L.ALPHA.get(strat), 'params': params, 'K': K_MAIN,
                                  'metrics': {k: {kk: (None if vv is None else round(vv, 6))
                                                  for kk, vv in v.items()} for k, v in m.items()}})
            print(f'  [table seed {seed}] {strat} ({time.time()-t0:.0f}s)', flush=True)
    with open(f'{OUT}/stratified_table.jsonl', 'w') as f:
        for l in lines:
            f.write(json.dumps(l) + '\n')

    # ---------- P-CAS1 / P-CAS1b / P-CAS2 ----------
    P = {'P-CAS1': {}, 'P-CAS1b': {}, 'P-CAS2': {}, 'P-CAS4': {}}
    for conv in L.CONVS:
        for strat in WIN:
            for seed in (0, 1, 2):
                cells = [ENT[(seed, strat, dr, conv)] for dr in range(5)]
                m1s = [metrics_all(e, C1) for e in cells]
                pt = {k: {'cas': round(float(np.mean([m['cas'][k] for m in m1s])), 4),
                          'oracle': round(float(np.mean([m['oracle'][k] for m in m1s])), 4)}
                      for k in ('pooled', 'std', 'F')}
                for k in pt:
                    pt[k]['diff'] = round(pt[k]['cas'] - pt[k]['oracle'], 4)
                for k, key in (('std', 'P-CAS1'), ('pooled', 'P-CAS1b')):
                    diffs = []
                    for c in L.boot_weights(L.EVAL, N_BOOT):
                        dd = []
                        for e in cells:
                            m = metrics_all(e, c)
                            if m['cas'][k] is not None and m['oracle'][k] is not None:
                                dd.append(m['cas'][k] - m['oracle'][k])
                        if dd:
                            diffs.append(float(np.mean(dd)))
                    d = np.array(diffs)
                    P[key].setdefault(f'{strat}|{conv}', {})[str(seed)] = {
                        'point': pt[k]['diff'], 'cas': pt[k]['cas'], 'oracle': pt[k]['oracle'],
                        'boot_mean': round(float(d.mean()), 4),
                        'ci95': [round(float(np.percentile(d, 2.5)), 4),
                                 round(float(np.percentile(d, 97.5)), 4)],
                        'frac_gt0': round(float((d > 0).mean()), 4)}
                Fm = {nm: float(np.mean([m[nm]['F'] for m in m1s])) for nm in NAMES}
                Sm = {nm: float(np.mean([m[nm]['std'] for m in m1s])) for nm in NAMES}
                P['P-CAS2'].setdefault(f'{strat}|{conv}', {})[str(seed)] = {
                    'F_table': {k: round(v, 4) for k, v in Fm.items()},
                    'std_table': {k: round(v, 4) for k, v in Sm.items()},
                    'CAS_ge_gate': bool(Fm['cas'] >= Fm['gate']),
                    'gate_ge_pool': bool(Fm['gate'] >= Fm['pool']),
                    'full_chain': bool(Fm['cas'] >= Fm['gate'] >= Fm['pool'])}
                print(f'  [{conv} {strat} s{seed}] F: ' + ' '.join(f'{k}={Fm[k]:.3f}' for k in NAMES)
                      + ' | std: ' + ' '.join(f'{k}={Sm[k]:.3f}' for k in NAMES), flush=True)
        for strat in EXTREME:
            for seed in (0, 1, 2):
                cells = [ENT[(seed, strat, dr, conv)] for dr in range(5)]
                m1s = [metrics_all(e, C1) for e in cells]
                P['P-CAS4'].setdefault(f'{strat}|{conv}', {})[str(seed)] = {
                    'F': {nm: round(float(np.mean([m[nm]['F'] for m in m1s])), 4) for nm in NAMES},
                    'std': {nm: round(float(np.mean([m[nm]['std'] for m in m1s])), 4) for nm in NAMES}}
            print(f'  [P-CAS4 {conv} {strat}] done ({time.time()-t0:.0f}s)', flush=True)

    verdict = {}
    for key in ('P-CAS1', 'P-CAS1b'):
        per_cell = {}
        for k, seeds in P[key].items():
            per_cell[k] = bool(all(seeds[str(s)]['ci95'][0] > 0 and seeds[str(s)]['point'] >= MIN_GAP
                                   for s in (0, 1, 2)))
        verdict[key] = {'all_cells_pass': bool(all(per_cell.values())),
                        'per_cell': per_cell, 'min_gap': MIN_GAP}
    verdict['P-CAS2'] = {'all_cells_pass': bool(all(all(seeds[str(s)]['full_chain'] for s in (0, 1, 2))
                                                    for seeds in P['P-CAS2'].values())),
                         'per_cell': {k: bool(all(seeds[str(s)]['full_chain'] for s in (0, 1, 2)))
                                      for k, seeds in P['P-CAS2'].items()},
                         'per_cell_CAS_ge_gate': {k: bool(all(seeds[str(s)]['CAS_ge_gate']
                                                             for s in (0, 1, 2)))
                                                  for k, seeds in P['P-CAS2'].items()}}

    # ---------- P-CAS3（原预测）+ P-CAS3b（桶分解） ----------
    r3 = {}
    for conv in L.CONVS:
        vs = [metrics_all(ENT[(s, 'S_a1.5', dr, conv)], C1)['cas']['std']
              for s in (0, 1, 2) for dr in range(5)]
        r3[conv] = {'CAS_std_at_alpha_star': round(float(np.mean(vs)), 4), 'threshold': 0.65,
                    'pass': bool(np.mean(vs) >= 0.65)}
    r3b = {}
    for seed in (0, 1, 2):
        bucket, wit = [], []
        for dr in range(5):
            case = L.load_case(seed, 'S_a1.5', dr, 'asimpl')
            sk = {d: np.clip((L.ranknorm(case[d]['deg']) * K_MAIN).astype(int), 0, K_MAIN - 1)
                  for d in L.EVAL}
            yb = {d: case[d]['y'] for d in L.EVAL}
            pk = L.pack({d: case[d]['r_learn'] for d in L.EVAL}, yb, L.EVAL, sk, K_MAIN)
            for d in L.EVAL:
                r = case[d]['r_learn']; y = case[d]['y']; deg = case[d]['deg']
                thr = np.quantile(deg, 0.5); sel = deg <= thr
                if y[sel].sum() == 0 or (1 - y[sel]).sum() == 0:
                    continue
                negs = np.sort(r[sel & (y == 0)]); pos = r[sel & (y == 1)]
                lo = np.searchsorted(negs, pos, 'left'); hi = np.searchsorted(negs, pos, 'right')
                bucket.append(float(((lo + 0.5 * (hi - lo)) / len(negs)).mean()))
            e0 = L.auc_from(pk['strata'][0], C1); e1 = L.auc_from(pk['strata'][1], C1)
            wit.append((e0 + e1) / 2)
        r3b[str(seed)] = {'bucket_lowdeg_auc': round(float(np.mean(bucket)), 4),
                          'within_strata_mean_auc_q0_q1': round(float(np.mean(wit)), 4),
                          'cross_term': round(float(np.mean(bucket) - np.mean(wit)), 4)}

    # ---------- M1 可操纵性 ----------
    m1 = {'learn_pooled_by_alpha': {}, 'learn_std_by_alpha': {}}
    for strat in L.STRATS:
        if strat not in L.ALPHA:
            continue
        pv = [l['metrics']['learn']['pooled'] for l in lines
              if l['strategy'] == strat and l['conv'] == 'asimpl']
        sv = [l['metrics']['learn']['std'] for l in lines
              if l['strategy'] == strat and l['conv'] == 'asimpl']
        m1['learn_pooled_by_alpha'][f'{L.ALPHA[strat]:+.2f}'] = {
            'mean': round(float(np.mean(pv)), 4), 'min': round(float(np.min(pv)), 4)}
        m1['learn_std_by_alpha'][f'{L.ALPHA[strat]:+.2f}'] = round(float(np.mean(sv)), 4)
    fl = {}
    for conv in L.CONVS:
        for k in ('pooled', 'std', 'F'):
            vals = []
            for seed in (0, 1, 2):
                vals.append(float(np.mean([metrics_all(ENT[(seed, 'S_a1.5', dr, conv)], C1)['gate'][k]
                                           - metrics_all(ENT[(seed, 'S_a1.5', dr, conv)],
                                                         C1)['degfall'][k] for dr in range(5)])))
            fl.setdefault(conv, {})[k] = round(float(np.mean(vals)), 4)
    m1['gate_minus_degfall_at_alpha_star'] = fl
    m1['learn_pooled_range_over_alpha_family'] = [
        round(min(v['min'] for v in m1['learn_pooled_by_alpha'].values()), 4),
        round(max(v['mean'] for v in m1['learn_pooled_by_alpha'].values()), 4)]

    # ---------- K=10 敏感性（报告型，seed0 窗口 asimpl） ----------
    k10 = {}
    for strat in WIN:
        vs = []
        for dr in range(5):
            params = ref[(0, strat, dr, 'asimpl')]['params']
            e = build_entries(0, strat, dr, 'asimpl', params, 10)
            m = metrics_all(e, C1)
            vs.append(m['cas'])
        k10[strat] = {'CAS_std_mean_K10': round(float(np.mean([v['std'] for v in vs])), 4),
                      'CAS_F_mean_K10': round(float(np.mean([v['F'] for v in vs])), 4)}

    out = {'experiment': 'E32', 'criteria_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cm)),
           'criteria_path': CRIT, 'verdict': verdict,
           'P-CAS1_primary_std': P['P-CAS1'], 'P-CAS1b_secondary_pooled': P['P-CAS1b'],
           'P-CAS2_F_ordering': P['P-CAS2'], 'P-CAS3_original_prediction': r3,
           'P-CAS3b_bucket_dissection': r3b, 'P-CAS4_extreme_domain': P['P-CAS4'],
           'M1_manipulability': m1, 'K10_sensitivity': k10,
           'n_cells': len(lines), 'elapsed_sec': round(time.time() - t0, 1)}
    json.dump(out, open(f'{OUT}/analysis_e32.json', 'w'), indent=1, ensure_ascii=False)
    open(f'{OUT}/DONE.sentinel', 'w').close()
    pm = os.path.getmtime(f'{OUT}/DONE.sentinel')
    print(f'== E32 DONE ({time.time()-t0:.0f}s) | criteria mtime < product mtime: {cm < pm}', flush=True)
    print('P-CAS1:', json.dumps(verdict['P-CAS1'], ensure_ascii=False), flush=True)
    print('P-CAS1b:', json.dumps(verdict['P-CAS1b'], ensure_ascii=False), flush=True)
    print('P-CAS2:', json.dumps(verdict['P-CAS2']['per_cell'], ensure_ascii=False), flush=True)
    print('P-CAS3:', json.dumps(r3, ensure_ascii=False), '| 3b:', json.dumps(r3b, ensure_ascii=False), flush=True)
    print('M1 range:', m1['learn_pooled_range_over_alpha_family'],
          json.dumps(fl, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
