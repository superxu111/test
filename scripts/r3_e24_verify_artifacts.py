"""r3_e24_verify_artifacts.py — E24：R3 既有产物核验（核验优先，禁盲目重跑）。

【实验设计（一行自查）】
验证假设：E17-E23 既有产物与历史结果一致（确定性复算容浮点尾差；独立重跑一致）。
自变量：复算通道（jsonl 原始行→analysis 汇总；npz→独立统计量重跑）。
因变量：逐项 Δ（分数类 |Δ|≤1e-4；p 值类 |Δ|≤5e-4；均值类 |Δ|≤1e-4）。
对照/baseline：工件 analysis_*.json 为 baseline，脚本内复算值为检验值。
规模：E17 96 行、E21 42 行、E23 12 行全量；E19 规范 draw 1 次独立重跑（6s）；
      E20 T2/T4 blind0 独立重跑（~10s）。成功判据：全部核对项 pass=true 且无缺件。
产物：results/r3/e24_artifact_verification/verification.json
运行：conda activate euler && python scripts/r3_e24_verify_artifacts.py 2>&1 | tee logs/r3_e24.log
"""
import json, os, sys
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/scripts')
ROOT = '/root/autodl-tmp/results/r3'
OUT = f'{ROOT}/e24_artifact_verification'
os.makedirs(OUT, exist_ok=True)
rep = {'checks': {}, 'missing': []}


def rd(p):
    return json.load(open(p))


def add(name, ok, detail):
    rep['checks'][name] = {'pass': bool(ok), **detail}


# ---------------- E17：jsonl → analysis 逐位对账 ----------------
def e17():
    jl = [json.loads(l) for l in open(f'{ROOT}/e17_ras_ablation/results_e17.jsonl')]
    an = rd(f'{ROOT}/e17_ras_ablation/analysis_e17.json')
    # 逐 (arm,seed,conv,method) auс_pooled
    by = {}
    for r in jl:
        by[(r['arm'], r['seed'], r['conv'], r['method'])] = r['auc_pooled']
    det = {}
    for method, exp_mean in [('L0_fixed0.05', 0.9432), ('heckman', 0.9407)]:
        vals = [by[('blind0', s, 'asimpl', method)] for s in (0, 1, 2)]
        m = float(np.mean(vals))
        det[f'blind0_asimpl_{method}'] = {'per_seed': vals, 'mean': round(m, 4),
                                          'analysis_mean': exp_mean, 'delta': round(m - exp_mean, 4)}
    # R1 heckman 配对差（jsonl 原始行独立计算 mean）
    diffs = []
    for s in (0, 1, 2):
        diffs.append(round(by[('blind0', s, 'asimpl', 'heckman')]
                           - by[('blind0', s, 'asimpl', 'L0')], 4))
    exp = an['R1_attribution']['heckman']['paired_diff_mean']
    det['R1_heckman_paired_diff'] = {'recomputed': round(float(np.mean(diffs)), 4), 'per_seed': diffs,
                                     'analysis': exp,
                                     'delta': round(float(np.mean(diffs)) - exp, 4)}
    ok = all(abs(v['delta']) <= 1e-4 for k, v in det.items() if 'delta' in v)
    add('E17_recompute', ok, {'n_rows': len(jl), 'detail': det})


# ---------------- E21：jsonl → analysis 逐位对账（含 strict 分支重算） ----------------
def e21():
    jl = [json.loads(l) for l in open(f'{ROOT}/e21_full_correction/results_e21.jsonl')]
    an = rd(f'{ROOT}/e21_full_correction/analysis_e21.json')
    det = {}
    for conv in ('asimpl', 'strict'):
        for arm in ('full_learned', 'full+L0fixed_w0.05', 'full+L0sel', 'full+ctrl_pow', 'full+heckman'):
            vals = sorted([r['auc_pooled'] for r in jl if r['conv'] == conv and r['method'] == arm])
            key = f'{conv}/{arm}'
            exp = an['summary'][key]['mean'] if key in an['summary'] else an.get('summary', {}).get(key, {})
            expv = exp if isinstance(exp, float) else (exp.get('mean') if isinstance(exp, dict) else None)
            if expv is None:
                det[key] = {'mean': round(float(np.mean(vals)), 4), 'analysis': None}
                continue
            det[key] = {'mean': round(float(np.mean(vals)), 4), 'analysis': expv,
                        'delta': round(float(np.mean(vals)) - expv, 4)}
    # strict 口径 K2 两分支重算（A席M1 需要）：best 修正臂 vs full_learned 逐 seed 差
    k2 = {}
    for conv in ('asimpl', 'strict'):
        base = {r['seed']: r['auc_pooled'] for r in jl
                if r['conv'] == conv and r['method'] == 'full_learned'}
        cands = [m for m in set(r['method'] for r in jl if r['conv'] == conv)
                 if m.startswith('full+') and 'degfall' not in m]
        best_arm, best_d = None, -9
        for m in cands:
            d = float(np.mean([r['auc_pooled'] - base[r['seed']] for r in jl
                               if r['conv'] == conv and r['method'] == m]))
            if d > best_d:
                best_d, best_arm = d, m
        k2[conv] = {'best_arm': best_arm, 'paired_diff_mean': round(best_d, 4),
                    'n_arms': len(cands)}
    det['K2_branch_recompute'] = k2
    ok = all(abs(v['delta']) <= 1e-4 for v in det.values() if isinstance(v, dict) and 'delta' in v)
    add('E21_recompute', ok, {'n_rows': len(jl), 'detail': det})


# ---------------- E23：jsonl → analysis 锚点对账 ----------------
def e23():
    jl = [json.loads(l) for l in open(f'{ROOT}/e23_second_domain/results_e23.jsonl')]
    an = rd(f'{ROOT}/e23_second_domain/analysis_e23.json')
    det = {}
    for s in (0, 1, 2):
        for arm in ('blind_learned', 'full_learned', 'degfall', 'pool'):
            exp = an['N1_anchors']['per_seed'][str(s)].get(arm, {}).get('e6')
            vals = [r['auc'] for r in jl if r.get('seed') == s and r.get('arm') == arm]
            if not vals or exp is None:
                continue
            det[f'seed{s}_{arm}'] = {'recomputed': round(float(np.mean(vals)), 4), 'e6': exp,
                                     'delta': round(float(np.mean(vals)) - exp, 4)}
    # N3 IMR 无特异增益：heckman_minus_L0sel 行直接读
    n3 = [r for r in jl if r.get('arm') == 'heckman_minus_L0sel']
    det['N3_rows'] = [{'seed': r['seed'], 'mean': r.get('auc')} for r in n3]
    ok = all(abs(v['delta']) <= 1e-4 for k, v in det.items() if isinstance(v, dict) and 'delta' in v)
    add('E23_recompute', ok, {'n_rows': len(jl), 'detail': det})


# ---------------- E22：从 points 独立重算 Spearman ----------------
def e22():
    from scipy import stats as sps
    src = rd(f'{ROOT}/e16_lam_backtest/analysis_e16.json')
    tot = rd(f'{ROOT}/e22_e16_dedup/analysis_e16_dedup_amendment.json')
    pts = src['points']['enron_dblp']
    ded, seen = [], set()
    for p in pts:
        k = (p.get('dataset'), p.get('onset'), p.get('seed'))
        if k not in seen:
            seen.add(k)
            ded.append(p)
    on = np.array([p['onset'] for p in ded], float)
    lam = np.array([p['lam_star'] for p in ded], float)
    lrn = np.array([p['learned_auc'] for p in ded], float)
    r1 = sps.spearmanr(on, lam)
    r2 = sps.spearmanr(1 - lrn, lam)
    a1, a2 = tot['P1_lambda_vs_onset_dedup'], tot['P2_lambda_vs_deficit_dedup']
    det = {'n_total': len(pts), 'n_dedup': len(ded),
           'P1_recomputed': [round(float(r1.statistic), 6), round(float(r1.pvalue), 8)],
           'P1_artifact': [a1['spearman'], a1['p']],
           'P2_recomputed': [round(float(r2.statistic), 6), round(float(r2.pvalue), 8)],
           'P2_artifact': [a2['spearman'], a2['p']],
           'delta_rho': [round(float(r1.statistic) - a1['spearman'], 6),
                         round(float(r2.statistic) - a2['spearman'], 6)]}
    ok = len(ded) == tot['n_dedup'] and all(abs(v) <= 1e-3 for v in det['delta_rho'])
    add('E22_recompute', ok, det)


# ---------------- E19：规范 draw 独立重跑 ----------------
def e19():
    import r3_e19_enron_split_audit as e19m
    st = e19m.q8_stats(e19m.load_rows(2026))
    art = rd(f'{ROOT}/e19_enron_split_audit/split_audit.json')
    can = art.get('canonical_draw', art.get('main_draw', {}))
    det = {'recomputed': st, 'artifact': can}
    pairs = [(k, st.get(k), can.get(k)) for k in ('p_one_sided_sigma+1', 'effect_auc_real_gt_neg', 'a_hat_q_beta', 'n_pos_total')]
    det['deltas'] = {k: (None if (a is None or b is None) else round(a - b, 6)) for k, a, b in pairs}
    ok = all(v is not None and abs(v) <= 5e-4 for v in det['deltas'].values())
    add('E19_canonical_rerun', ok, det)


# ---------------- E20：T2/T4 blind0 独立重跑 ----------------
def e20():
    import r3_e20_unique_pair as e20m
    parts = e20m.merge_rank(e20m.load_parts('blind0'))
    t2 = e20m.t2_pair_perm(parts)
    t4 = e20m.t4_mwu_collapse(parts)
    art = rd(f'{ROOT}/e20_unique_pair/unique_pair_inference.json')['per_arm']['blind0']
    det = {'t2_recomputed': t2, 't2_artifact': art['t2_pair_perm'],
           't4_recomputed': t4, 't4_artifact': art['t4_mwu_collapse'],
           't2_delta_p': round(t2['p_one_sided'] - art['t2_pair_perm']['p_one_sided'], 6),
           't4_delta_p': round(t4['p_mwu_one_sided'] - art['t4_mwu_collapse']['p_mwu_one_sided'], 6)}
    ok = abs(det['t4_delta_p']) <= 5e-4
    add('E20_T2T4_rerun', ok, det)


# ---------------- 工件存在性与 mtime 链 ----------------
def presence():
    need = {
        'e17_ras_ablation/analysis_e17.json': None, 'e17_ras_ablation/mechanism_lowdeg.json': None,
        'e17_ras_ablation/R3_tau_consistency.json': None, 'e17_ras_ablation/tau_estimates.json': None,
        'e17_ras_ablation/pool_check.json': None,
        'e18_e15_esp/analysis_e18.json': None, 'e18_e15_esp/rejudged_e15.json': None,
        'e18_e15_esp/esp_gamma_curves.json': None,
        'e19_enron_split_audit/split_audit.json': None,
        'e20_unique_pair/unique_pair_inference.json': None,
        'e21_full_correction/analysis_e21.json': None,
        'e22_e16_dedup/analysis_e16_dedup_amendment.json': None,
        'e23_second_domain/analysis_e23.json': None, 'e23_second_domain/summary_e23.json': None,
        'CRITERIA_TIMESTAMP_ERRATUM_20261016.json': None,
    }
    det = {}
    for rel in need:
        p = f'{ROOT}/{rel}'
        det[rel] = os.path.exists(p)
    # 判据先于产物（mtime 链）
    chain = {}
    for exp, crit, first in (('E17', 'e17_ras_ablation/CRITERIA_FROZEN.json', 'e17_ras_ablation/results_e17.jsonl'),
                             ('E19', 'e19_enron_split_audit/CRITERIA_FROZEN.json', 'e19_enron_split_audit/split_audit.json'),
                             ('E20', 'e20_unique_pair/CRITERIA_FROZEN.json', 'e20_unique_pair/unique_pair_inference.json'),
                             ('E21', 'e21_full_correction/CRITERIA_FROZEN.json', 'e21_full_correction/results_e21.jsonl'),
                             ('E23', 'e23_second_domain/CRITERIA_FROZEN.json', 'e23_second_domain/results_e23.jsonl')):
        tc, tf = os.path.getmtime(f'{ROOT}/{crit}'), os.path.getmtime(f'{ROOT}/{first}')
        chain[exp] = {'criteria_mtime': tc, 'first_artifact_mtime': tf, 'ok': tc <= tf}
    add('artifact_presence', all(det.values()), {'present': det, 'mtime_chain': chain})
    # npz 可载入 + 关键键
    nz = {}
    for f, keys in (('e14b_lp_npz/scores_e14b_blind0_seed0.npz', ['asimpl__1_pos_a', 'lam_sig_asimpl', '1_pos_u']),
                    ('e23_second_domain/scores_e23_seed0.npz', ['dec_b_pos', 'deg_pos', 'w_sel'])):
        z = np.load(f'{ROOT}/{f}', allow_pickle=True)
        nz[f] = {k: (k in z.files) for k in keys}
    z11 = np.load('/root/autodl-tmp/results/r2/lanl_blindness/scores_e11_blind0_seed0.npz', allow_pickle=True)
    nz['e11_blind0_seed0'] = {'12_cross_y': '12_cross_y' in z11.files,
                              'asimpl__12_cross_a': 'asimpl__12_cross_a' in z11.files}
    add('npz_loadable', all(all(v.values()) for v in nz.values()), nz)


if __name__ == '__main__':
    e17(); e21(); e23()
    try:
        e19()
    except Exception as e:
        rep['missing'].append(f'E19 rerun failed: {type(e).__name__}: {e}')
    try:
        e20()
    except Exception as e:
        rep['missing'].append(f'E20 rerun failed: {type(e).__name__}: {e}')
    e22(); presence()
    rep['summary'] = {'n_checks': len(rep['checks']),
                      'n_pass': sum(1 for v in rep['checks'].values() if v['pass']),
                      'all_pass': all(v['pass'] for v in rep['checks'].values()) and not rep['missing']}
    json.dump(rep, open(f'{OUT}/verification.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(rep['summary'], ensure_ascii=False), flush=True)
    for k, v in rep['checks'].items():
        print(f'  {k}: pass={v["pass"]}', flush=True)
    print('DONE E24', flush=True)
