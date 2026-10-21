"""r3_e33_a4prime_adaptive.py — E33：A4′ 信息集对称自适应白盒审计（免训臂）。

判据：results/r3/e33_a4prime_adaptive/criteria_frozen.json（先冻结后跑数；打印判据 mtime 与产物 mtime）
运行：conda activate euler && setsid nohup python scripts/r3_e33_a4prime_adaptive.py > logs/r3_e33.log 2>&1 &
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/scripts')
import r3_e31_lib as L

OUT = '/root/autodl-tmp/results/r3/e33_a4prime_adaptive'
CRIT = f'{OUT}/criteria_frozen.json'
NAMES = ('learn', 'degfall', 'gate', 'pool', 'cas')
N_BOOT = 1000
K = 5
C1 = np.ones(len(L.EVAL))
TARGETS = [('learn', None, 'learn')] + \
          [(f'gate_{w}', w, 'gate') for w in L.W_GRID] + \
          [('cas', None, 'cas'), ('e25_A4ref', None, 'e25ref')]


def build(target, seed, dr, conv):
    """返回 (case, params, case 的 pack 字典)。target 语义：learn/gate_w/cas/e25_A4ref。"""
    name, w_att, kind = target
    if kind == 'e25ref':          # E25 原 A4 链（W_BR=0.05 硬编码），用于失配基线复现
        case = L.raw_case(seed, 'A4_br_gate', dr, conv, days=L.ALLD)
    else:
        case = L.case_extra_br(seed, dr, L.ALLD, w_att=w_att, target=kind)
    p = L.select_params_val(case, conv)
    yb = {d: case[d]['y'] for d in L.EVAL}
    sk = {d: np.clip((L.ranknorm(L.deg_day(case, d)) * K).astype(int), 0, K - 1) for d in L.EVAL}
    pk = {}
    for nm in NAMES:
        rk = {d: L.ranks_day(case, d, (nm,), p, conv)[nm] for d in L.EVAL}
        pk[nm] = L.pack(rk, yb, L.EVAL, sk, K)
    return case, p, pk


def main():
    t0 = time.time()
    cm = os.path.getmtime(CRIT)
    print(f'== E33 | criteria mtime={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cm))} '
          f'| now={time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    ref = {(r['seed'], r['strategy'], r['draw'], r['conv']): r
           for r in (json.loads(l) for l in open(f'{L.BED}/results_e25.jsonl'))}

    STORE = {}
    rows = []
    for seed in (0, 1, 2):
        for dr in range(5):
            for conv in L.CONVS:
                for target in TARGETS:
                    case, p, pk = build(target, seed, dr, conv)
                    m = L.metrics_from_pk(pk, C1)
                    STORE[(seed, dr, conv, target[0])] = {'pk': pk, 'params': p}
                    rows.append({'seed': seed, 'draw': dr, 'conv': conv, 'target': target[0],
                                 'attacker_w': target[1], 'defender_params': p,
                                 'medpct': round(float(np.median(np.concatenate(
                                     [L.pct_in(case[d]['deg_pos'], np.sort(case[d]['deg_neg']))
                                      for d in L.EVAL]))), 4),
                                 'metrics': {k: {kk: m[k][kk] for kk in ('pooled', 'std', 'F')}
                                             for k in NAMES}})
                print(f'  [seed {seed} dr {dr} {conv}] gate pooled by target: ' +
                      ' '.join(f'{t[0]}={np.mean([r["metrics"]["gate"]["pooled"] for r in rows if r["seed"]==seed and r["draw"]==dr and r["conv"]==conv and r["target"]==t[0]]):.3f}'
                               for t in TARGETS) + f' ({time.time()-t0:.0f}s)', flush=True)

    def gate_pooled(key):
        pk = STORE[key]['pk']['gate']['glob']
        return float(pk[0].sum() / (pk[1].sum() * pk[2].sum()))

    # ---- J1：失配量化（配对日块 bootstrap，逐 draw 平均） ----
    def paired(seed, t1, t2, conv='asimpl', nm='gate', k='pooled'):
        diffs = []
        for c in L.boot_weights(L.EVAL, N_BOOT):
            dd = []
            for dr in range(5):
                e1 = STORE[(seed, dr, conv, t1)]['pk'][nm]
                e2 = STORE[(seed, dr, conv, t2)]['pk'][nm]
                if k == 'std':
                    v1 = L.metrics_from_pk({'x': e1}, c, names=('x',))['x']['std']
                    v2 = L.metrics_from_pk({'x': e2}, c, names=('x',))['x']['std']
                else:
                    v1 = L.auc_from(e1['glob'], c); v2 = L.auc_from(e2['glob'], c)
                if v1 is None or v2 is None:
                    continue
                dd.append(v1 - v2)
            if dd:
                diffs.append(float(np.mean(dd)))
        d = np.array(diffs)
        return {'mean': round(float(d.mean()), 4),
                'ci95': [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)],
                'frac_lt0': round(float((d < 0).mean()), 4)}

    j1 = {}
    for seed in (0, 1, 2):
        j1[str(seed)] = {
            'A4prime_gt0.5_vs_e25_A4ref': paired(seed, 'gate_0.5', 'e25_A4ref'),
            'A4prime_gt0.5_vs_A4prime_gt0.05': paired(seed, 'gate_0.5', 'gate_0.05'),
            'point_gate_pooled': {t[0]: round(float(np.mean([gate_pooled((seed, dr, 'asimpl', t[0]))
                                                             for dr in range(5)])), 4) for t in TARGETS}}
    # 复现校验：e25_A4ref 与 E25 记录逐格对比
    xc = {}
    for seed in (0, 1, 2):
        for conv in L.CONVS:
            got = [round(float(gate_pooled((seed, dr, conv, 'e25_A4ref')).__round__(4)), 4) for dr in range(5)]
            exp = [ref[(seed, 'A4_br_gate', dr, conv)]['auc']['gate'] for dr in range(5)]
            ok = all(abs(g - e) < 2e-3 for g, e in zip(got, exp))  # E25 工件 4 位小数
            xc[f'{seed}_{conv}'] = {'got': got, 'exp': exp, 'match': bool(ok)}
    print('  [复现校验 e25_A4ref]', json.dumps({k: v['match'] for k, v in xc.items()}), flush=True)

    # ---- J2/J4：自适应表 ----
    tab = {}
    for target in TARGETS:
        name = target[0]
        e = {'gate_pooled': [], 'gate_std': [], 'gate_F': [], 'degfall_pooled': [],
             'cas_pooled': [], 'cas_std': [], 'cas_F': [], 'defender_gate_w': [], 'medpct': []}
        for r in rows:
            if r['target'] != name or r['conv'] != 'asimpl':
                continue
            e['gate_pooled'].append(r['metrics']['gate']['pooled'])
            e['gate_std'].append(r['metrics']['gate']['std'])
            e['gate_F'].append(r['metrics']['gate']['F'])
            e['degfall_pooled'].append(r['metrics']['degfall']['pooled'])
            e['cas_pooled'].append(r['metrics']['cas']['pooled'])
            e['cas_std'].append(r['metrics']['cas']['std'])
            e['cas_F'].append(r['metrics']['cas']['F'])
            e['defender_gate_w'].append(r['defender_params']['gate_w'])
            e['medpct'].append(r['medpct'])
        from scipy import stats as sps
        tab[name] = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in e.items() if k != 'defender_gate_w'}
        tab[name]['defender_gate_w_mode'] = float(sps.mode(e['defender_gate_w'], keepdims=False).mode)
        tab[name]['defender_gate_w_counts'] = {str(float(w)): int(e['defender_gate_w'].count(w))
                                              for w in sorted(set(e['defender_gate_w']))}
    # 最强自适应 = min over targets of gate pooled
    adaptive_best = min((v['gate_pooled'], k) for k, v in tab.items() if v['gate_pooled'] is not None)

    # ---- J3：CAS 序在自适应下 ----
    j3 = {}
    for target in TARGETS:
        name = target[0]
        Fs = {'cas': [], 'gate': []}
        for r in rows:
            if r['target'] == name and r['conv'] == 'asimpl':
                Fs['cas'].append(r['metrics']['cas']['F']); Fs['gate'].append(r['metrics']['gate']['F'])
        if Fs['cas']:
            j3[name] = {'F_cas_mean': round(float(np.mean(Fs['cas'])), 4),
                        'F_gate_mean': round(float(np.mean(Fs['gate'])), 4),
                        'cas_ge_gate': bool(np.mean(Fs['cas']) >= np.mean(Fs['gate']))}

    out = {'experiment': 'E33', 'criteria_mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cm)),
           'criteria_path': CRIT,
           'defect_confirmed': {'A4_br_gate_defender_gate_w_asimpl_all_0.5': True,
                                'recorded_attacker_W_BR': 0.05},
           'xcheck_e25_A4ref_vs_record': xc,
           'J1_mismatch_quantified': j1,
           'J2_J4_adaptive_table': tab,
           'adaptive_best_min_gate_pooled': {'value': round(float(adaptive_best[0]), 4),
                                             'target': adaptive_best[1]},
           'J3_cas_ordering_under_adaptive': j3,
           'rows': rows, 'n_rows': len(rows),
           'elapsed_sec': round(time.time() - t0, 1)}
    json.dump(out, open(f'{OUT}/analysis_e33.json', 'w'), indent=1, ensure_ascii=False)
    with open(f'{OUT}/results_e33.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    open(f'{OUT}/DONE.sentinel', 'w').close()
    pm = os.path.getmtime(f'{OUT}/DONE.sentinel')
    print(f'== E33 DONE ({time.time()-t0:.0f}s) | criteria mtime < product mtime: {cm < pm}', flush=True)
    print('J1:', json.dumps({k: {'v_e25ref': v['A4prime_gt0.5_vs_e25_A4ref'],
                                 'v_w005': v['A4prime_gt0.5_vs_A4prime_gt0.05']} for k, v in j1.items()},
                            ensure_ascii=False), flush=True)
    print('adaptive best:', json.dumps(out['adaptive_best_min_gate_pooled'], ensure_ascii=False), flush=True)
    print('J3:', json.dumps(j3, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
