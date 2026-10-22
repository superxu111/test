"""r2_e8_strictvis.py — E8 严格可见性口径对称化重评（R2b 创新实验，判据见
results/r2/e8_strictvis/CRITERIA_FROZEN.json，先冻结后跑数，P17）。

回应该双审两席同判（A席M4+B席M2②）：FreeDyG 官方 find_neighbors_before 严格早于查询时刻
（/tmp/freedyg_readonly/utils/utils.py:130），Euler 臂评估编码含查询快照全图——可见性口径
系统性利于 Euler。本实验把「可见性口径」变成受控因子：
  as-implemented：Euler z=zs[t]（r1_e6_cal.py:296-306 同法）、degfall 全流累计度
                  （r1_e6_cal.py:177-181 同法，含查询日与未来日边——如实披露）、
                  EdgeBank 全流 memory（r2_t4_freedyp.py:180 同法）。
  strict（P17）：Euler z=zs[t−1]（GRU 只吞入快照 0..t−1，同模型同权重仅换编码截断）、
                degfall 用快照 0..t−1 累计度、EdgeBank 用快照 0..t−1 累计 memory。
训练完全复用 r1_e6_cal.train_model 协议（同 RNG 域）；评估集=同 setup() 固定跨界集（n_pos=26）。
FreeDyG 构造上严格历史，复用既有 npz 双口径同值，不重跑。
产出：results/r2/e8_strictvis/results_e8.json + scores_e8_*.npz（双口径逐日分数，免重训复分析）。

E9 扩展（R2c，P5/B席M4）：剂量谱 strict 臂——onset∈{1,3,7}×3 seed。参数化两点：
  --ref-dir  λ*/确定性参照 jsonl 目录（onset 0/5 恒取 e6_cal 冻结源；onset∈{1,3,7} 取 e7_dose）
  条件名 blindN 直接编码 onset=N（blind1/blind3/blind7）；输出走 --results-name/--npz-prefix 新文件不覆盖 E8 旧件。
判据见同目录 CRITERIA_E9_DOSE.json（先冻结后跑数）。
版本：v1（E8 原行为）归档于 scripts/r2_e8_strictvis_v1_R2c.py；默认参数下行为与 v1 逐位一致。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '/root/autodl-tmp/Euler-main')
sys.path.insert(0, '/root/autodl-tmp/Euler-main/benchmarks')
sys.path.insert(0, '/root/autodl-tmp/scripts')
from r1_e6_cal import setup, train_model, ranknorm, ranknorm_joint  # noqa: E402

OUT = '/root/autodl-tmp/results/r2/e8_strictvis'
FD = '/root/autodl-tmp/results/r2/t4_freedyp'
E6 = '/root/autodl-tmp/results/r1/e6_cal'
LAB = np.concatenate([np.ones(26), np.zeros(26)])  # 固定评估集标签（26 正 26 负，行序=评估日序）


def cum_degree(eis, N):
    """deg[d] = 快照 0..d 的累计度（含 d）；strict 在查询日 t 用 deg[t-1]"""
    deg = torch.zeros(N)
    out = []
    for t in range(len(eis)):
        for u, v in eis[t].t().tolist():
            deg[u] += 1
            deg[v] += 1
        out.append(deg.clone())
    return out  # len=T


def score_euler_dual(model, eis_arm, dom, day_degs):
    """同一训练后模型，双口径打分。返回 {(conv, tag): (st_d, sf_d)}，conv∈{asimpl,strict}"""
    x_eye = torch.eye(dom['N'])
    with torch.no_grad():
        zs = model(x_eye, eis_arm, lambda i: eis_arm[i])
    res = {}
    for conv in ('asimpl', 'strict'):
        st_d, sf_d = [], []
        for pos, neg in zip(dom['eval_pos'], dom['eval_neg']):
            if pos[0].size(0) == 0:
                st_d.append(np.array([])); sf_d.append(np.array([]))
                continue
            t = pos[2]
            z = zs[t] if conv == 'asimpl' else zs[t - 1]
            st_d.append(model.decode(pos[0], pos[1], z).numpy())
            sf_d.append(model.decode(neg[0], neg[1], z).numpy())
        res[('euler', conv)] = (st_d, sf_d)
    # degfall 双口径：asimpl=全流累计度（r1_e6_cal 原法，含未来日）；strict=0..t-1 累计度
    for conv in ('asimpl', 'strict'):
        st_d, sf_d = [], []
        for pos, neg in zip(dom['eval_pos'], dom['eval_neg']):
            if pos[0].size(0) == 0:
                st_d.append(np.array([])); sf_d.append(np.array([]))
                continue
            t = pos[2]
            deg = dom['deg_blind'] if conv == 'asimpl' else day_degs[t - 1]
            for e_, acc in ((pos, st_d), (neg, sf_d)):
                u, v = e_[0].numpy(), e_[1].numpy()
                vis = np.where(dom['Smask'][u].numpy(), v, u)
                acc.append(np.log1p(deg.numpy()[vis]))
        res[('degfall', conv)] = (st_d, sf_d)
    return res


def edgebank_dual(eis_stream, dom):
    """EdgeBank 双口径：asimpl=全流 memory；strict=查询日 t 前累计 memory。无向对称化同 T4。"""
    ev_days = []
    for t, e in enumerate(eis_stream):
        pairs = {(int(u), int(v)) for u, v in e.t().tolist()}
        ev_days.append(pairs)
    full_mem = set().union(*ev_days) if ev_days else set()
    full_mem |= {(v, u) for (u, v) in full_mem}
    res = {}
    for conv in ('asimpl', 'strict'):
        st_d, sf_d = [], []
        for pos, neg in zip(dom['eval_pos'], dom['eval_neg']):
            if pos[0].size(0) == 0:
                st_d.append(np.array([])); sf_d.append(np.array([]))
                continue
            t = pos[2]
            if conv == 'asimpl':
                mem = full_mem
            else:
                mem = set().union(*ev_days[:t]) if t > 0 else set()
                mem |= {(v, u) for (u, v) in mem}
            sc = np.array([1.0 if (int(a), int(b)) in mem else 0.0
                           for a, b in zip(pos[0].numpy(), pos[1].numpy())])
            sfn = np.array([1.0 if (int(a), int(b)) in mem else 0.0
                            for a, b in zip(neg[0].numpy(), neg[1].numpy())])
            st_d.append(sc); sf_d.append(sfn)
        res[conv] = (st_d, sf_d)
    return res


def flat(st_d, sf_d):
    return (np.concatenate([d for d in st_d if len(d)]),
            np.concatenate([d for d in sf_d if len(d)]))


def auc_of(st, sf):
    return float(roc_auc_score(LAB, np.concatenate([st, sf])))


def ci_strict_minus_asimpl(st_s, sf_s, st_a, sf_a, seed):
    """strict − asimpl 配对 bootstrap CI95（10000 次）"""
    from r1_e6_cal import paired_bootstrap_diff
    return paired_bootstrap_diff(st_s, sf_s, st_a, sf_a, n_boot=10000, seed=seed)


def load_lam_star(ref_dir, onset):
    vals = {}
    for line in open(f'{ref_dir}/results_enron10_community_0.4_onset{onset}.jsonl'):
        r = json.loads(line)
        vals[r['seed']] = float(r['lam_star'])
    return vals


def load_freedyg(cond, seed):
    z = np.load(f'{FD}/scores_enron10_community_0.4_{cond}_freedyg_seed{seed}.npz')
    days = sorted({int(k.rsplit('_', 1)[-1]) for k in z.files if k.startswith('freedyg_st_')})
    return (np.concatenate([z[f'freedyg_st_{d}'] for d in days]),
            np.concatenate([z[f'freedyg_sf_{d}'] for d in days]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--zdim', type=int, default=16)
    ap.add_argument('--test-ts', type=int, default=3)
    ap.add_argument('--dataset', default='enron10')
    ap.add_argument('--mode', default='community_0.4')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--conditions', default='blind5,blind0,full')
    ap.add_argument('--store-under', default='euler',
                    help='结果存储键：euler=主臂（含确定性断言）；euler_lowbudget=C5 低预算臂'
                         '（lr=1e-4/epochs≤100/patience=20 对齐 FreeDyG 官方预算，无确定性断言）')
    ap.add_argument('--onset', type=int, default=5,
                    help='E9 参数化②：blind 缺省条件的 onset（条件名 blindN 时被 N 覆盖）')
    ap.add_argument('--ref-dir', default='',
                    help='E9 参数化①：onset∉{0,5} 时 λ*/确定性参照 jsonl 所在目录（如 e7_dose）；'
                         'onset∈{0,5} 恒用 e6_cal（E8 冻结 λ* 源不变）')
    ap.add_argument('--results-name', default='results_e8.json',
                    help='结果 json 文件名（E9 用 results_e9_dose_strict.json，不覆盖 E8 旧件）')
    ap.add_argument('--npz-prefix', default='scores_e8',
                    help='npz 文件名前缀（E9 用 scores_e9）')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(4)

    outp = os.path.join(OUT, args.results_name)
    results = json.load(open(outp)) if os.path.exists(outp) else {}
    seeds = [int(s) for s in args.seeds.split(',')]
    conds = args.conditions.split(',')
    cond_idx = {'full': 0, 'blind0': 1, 'blind5': 2}
    for c in conds:  # E9：blindN 条件名的 bootstrap 种子索引（10+N 与 E8 的 8000/8300/8600/8900/9500 系无碰撞）
        if c not in cond_idx and c.startswith('blind') and c[5:].isdigit():
            cond_idx[c] = 10 + int(c[5:])

    def ref_dir_of(onset):
        return E6 if onset in (0, 5) else (args.ref_dir or E6)

    lam_tabs = {}

    def lam_table(onset):
        if onset not in lam_tabs:
            lam_tabs[onset] = load_lam_star(ref_dir_of(onset), onset)
        return lam_tabs[onset]

    for cond in conds:
        # E9 参数化②：条件名 blindN 编码 onset=N；full/blind0 恒 onset0；其余（含旧名 blind5）取 --onset
        if cond in ('full', 'blind0'):
            onset = 0
        elif cond.startswith('blind') and cond[5:].isdigit():
            onset = int(cond[5:])
        else:
            onset = args.onset
        torch.manual_seed(2026)  # 与 r1_e6_cal.main:248 同序，保证与 e6_cal/e7_dose 逐位同域
        dom = setup(args.mode, onset, args)
        N, T, end_tr = dom['N'], dom['T'], dom['end_tr']
        assert int(sum(p[0].size(0) for p in dom['eval_pos'])) == 26, '评估集同源断言失败'
        day_degs = cum_degree(dom['eis_blind'], N)  # strict degfall 用
        arm_eis = dom['eis_full'] if cond == 'full' else dom['eis_blind']
        arm_tr = dom['tr_full'] if cond == 'full' else dom['tr_blind']
        eb = edgebank_dual(arm_eis, dom)  # EdgeBank 双口径（无训练）
        results.setdefault(cond, {})
        store_under = args.store_under
        for seed in seeds:
            if str(seed) in results[cond].get(store_under, {}):
                continue
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = train_model('full' if cond == 'full' else 'blind',
                                arm_eis, arm_tr, N, torch.eye(N), end_tr, args, seed)
            scored = score_euler_dual(model, arm_eis, dom, day_degs)
            rec = {'sec': round(time.time() - t0, 1), 'auc': {}, 'ci': {}, 'edgebank': {},
                   'hyperparams': {'lr': args.lr, 'epochs': args.epochs, 'patience': args.patience}}
            # 确定性 sanity：asimpl AUC vs 参照 jsonl 现值（|Δ|≤0.01）——仅主预算臂（低预算臂协议不同无对照）
            ref_tag = 'full' if cond == 'full' else 'blind'
            ref = [json.loads(l) for l in open(
                f'{ref_dir_of(onset)}/results_enron10_community_0.4_onset{onset}.jsonl')][seed][ref_tag]['auc']
            st_a, sf_a = flat(*scored[('euler', 'asimpl')])
            rec['determinism_check'] = {'ref_e6cal': ref, 'rerun_asimpl': round(auc_of(st_a, sf_a), 4),
                                        'abs_dev': round(abs(ref - auc_of(st_a, sf_a)), 4)}
            if store_under == 'euler':
                assert rec['determinism_check']['abs_dev'] <= 0.01, \
                    f'确定性超限 {cond} seed{seed}: {rec["determinism_check"]}'
            # 双口径 AUC + 可见性主效应 CI
            for tag, conv in (('euler', 'asimpl'), ('euler', 'strict'),
                              ('degfall', 'asimpl'), ('degfall', 'strict')):
                st, sf = flat(*scored[(tag, conv)])
                rec['auc'][f'{tag}_{conv}'] = round(auc_of(st, sf), 4)
            for tag in ('euler', 'degfall'):
                st_s, sf_s = flat(*scored[(tag, 'strict')])
                ci = ci_strict_minus_asimpl(st_s, sf_s, *flat(*scored[(tag, 'asimpl')]),
                                            seed=8000 + 100 * cond_idx[cond] + 10 * seed)
                rec['ci'][f'{tag}_strict_minus_asimpl'] = {
                    'mean': round(ci['mean'], 4), 'ci95': ci['ci95'], 'p_gt0': round(ci['p_gt0'], 4)}
            # pool 双口径（λ* 冻结自 as-implemented 选择；E9 onset∈{1,3,7} 冻结源=e7_dose jsonl）
            lam = lam_table(onset)[seed]
            rec['lam_star_frozen'] = lam
            pool_scores = {}
            for conv in ('asimpl', 'strict'):
                stg, sfg = flat(*scored[('degfall', conv)])
                stl, sfl = flat(*scored[('euler', conv)])
                pr_p, pr_f = ranknorm_joint(stg, sfg)
                lr_p, lr_f = ranknorm_joint(stl, sfl)
                pool_scores[conv] = (lam * pr_p + (1 - lam) * lr_p, lam * pr_f + (1 - lam) * lr_f)
                rec['auc'][f'pool_{conv}'] = round(auc_of(*pool_scores[conv]), 4)
            ci = ci_strict_minus_asimpl(*pool_scores['strict'], *pool_scores['asimpl'],
                                        seed=8300 + 10 * seed)
            rec['ci']['pool_strict_minus_asimpl'] = {'mean': round(ci['mean'], 4),
                                                     'ci95': ci['ci95'], 'p_gt0': round(ci['p_gt0'], 4)}
            # EdgeBank 双口径
            for conv in ('asimpl', 'strict'):
                st, sf = flat(*eb[conv])
                rec['edgebank'][conv] = {
                    'auc': round(auc_of(st, sf), 4),
                    'ap': round(float(average_precision_score(LAB, np.concatenate([st, sf]))), 4)}
            st_s, sf_s = flat(*eb['strict']); st_a2, sf_a2 = flat(*eb['asimpl'])
            ci = ci_strict_minus_asimpl(st_s, sf_s, st_a2, sf_a2, seed=8600 + 10 * seed)
            rec['edgebank']['ci_strict_minus_asimpl'] = {'mean': round(ci['mean'], 4),
                                                         'ci95': ci['ci95']}
            # npz 落盘（双口径逐日，免重训复分析）
            npz_kw = {}
            for (tag, conv), (st_d, sf_d) in scored.items():
                for d, s in enumerate(st_d):
                    npz_kw[f'{tag}_{conv}_st_{d}'] = s
                for d, s in enumerate(sf_d):
                    npz_kw[f'{tag}_{conv}_sf_{d}'] = s
            np.savez_compressed(os.path.join(
                OUT, f'{args.npz_prefix}_{cond}_seed{seed}{"" if store_under == "euler" else "_lowbudget"}.npz'),
                **npz_kw)
            results[cond].setdefault(store_under, {})[str(seed)] = rec
            print(f'[{cond}/{store_under} seed{seed}] ' + json.dumps(rec['auc']) +
                  f' det={rec["determinism_check"]["abs_dev"]} ({rec["sec"]}s)', flush=True)
            json.dump(results, open(outp, 'w'), indent=1, ensure_ascii=False)

        # ---- C3：FreeDyG（构造严格）vs Euler strict（口径对称化后的 T4 重读）----
        if cond == 'blind5' and store_under == 'euler' and \
                all(str(s) in results[cond].get('euler', {}) for s in seeds):
            sym = {}
            for seed in seeds:
                f_st, f_sf = load_freedyg('blind5', seed)
                z = np.load(f'{OUT}/scores_e8_blind5_seed{seed}.npz')
                days = sorted({int(k.rsplit('_', 1)[-1]) for k in z.files
                               if k.startswith('euler_strict_st_')})
                e_st = np.concatenate([z[f'euler_strict_st_{d}'] for d in days])
                e_sf = np.concatenate([z[f'euler_strict_sf_{d}'] for d in days])
                from r1_e6_cal import paired_bootstrap_diff
                ci = paired_bootstrap_diff(f_st, f_sf, e_st, e_sf, n_boot=10000, seed=8900 + seed)
                sym[seed] = {'gap_point': round(auc_of(f_st, f_sf) - auc_of(e_st, e_sf), 4),
                             'ci95': ci['ci95'], 'p_gt0': round(ci['p_gt0'], 4)}
            results[cond]['freedyg_minus_euler_strict'] = sym
            json.dump(results, open(outp, 'w'), indent=1, ensure_ascii=False)
            print('[C3] freedyg−euler_strict', json.dumps(sym), flush=True)

        # ---- C5：低预算臂 vs FreeDyG（预算敏感性，B席M2①）----
        if cond == 'blind5' and store_under == 'euler_lowbudget' and \
                all(str(s) in results[cond].get(store_under, {}) for s in seeds):
            from r1_e6_cal import paired_bootstrap_diff
            sens = {}
            for seed in seeds:
                f_st, f_sf = load_freedyg('blind5', seed)
                z = np.load(f'{OUT}/scores_e8_blind5_seed{seed}_lowbudget.npz')
                row = {}
                for conv in ('asimpl', 'strict'):
                    days = sorted({int(k.rsplit('_', 1)[-1]) for k in z.files
                                   if k.startswith(f'euler_{conv}_st_')})
                    e_st = np.concatenate([z[f'euler_{conv}_st_{d}'] for d in days])
                    e_sf = np.concatenate([z[f'euler_{conv}_sf_{d}'] for d in days])
                    ci = paired_bootstrap_diff(f_st, f_sf, e_st, e_sf, n_boot=10000,
                                               seed=9500 + 10 * seed)
                    row[conv] = {'gap_point': round(auc_of(f_st, f_sf) - auc_of(e_st, e_sf), 4),
                                 'ci95': ci['ci95'], 'p_gt0': round(ci['p_gt0'], 4)}
                sens[seed] = row
            results[cond]['freedyg_minus_euler_lowbudget'] = sens
            json.dump(results, open(outp, 'w'), indent=1, ensure_ascii=False)
            print('[C5] freedyg−euler_lowbudget', json.dumps(sens), flush=True)
    print('E8_CONDITIONS_DONE', conds, flush=True)


if __name__ == '__main__':
    main()
