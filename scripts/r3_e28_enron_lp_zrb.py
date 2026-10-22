"""r3_e28_enron_lp_zrb.py — E28：W8 enron10 侧 LP 语义零回退表（E16b 同构；纯工件再表式化）。"""
import json, os
ROOT = '/root/autodl-tmp/results/r3'
OUT = f'{ROOT}/e28_enron_lp_zrb'
rows = [json.loads(l) for l in open(f'{ROOT}/e23_second_domain/results_e23.jsonl')]
arms = ['blind_learned', 'degfall', 'degfall_causal', 'L0fixed_w-0.05', 'L0sel', 'heckman', 'pool']
tab = {}
for a in arms:
    vals = [r['auc'] for r in rows if r.get('arm') == a]
    if vals:
        tab[a] = {'per_seed': vals, 'mean': round(sum(vals) / len(vals), 4)}
pool_mean = tab['pool']['mean']
rep = {'table': tab, 'pool_mean': pool_mean,
       'zero_rollback': {a: {'max_abs_delta_vs_pool': round(max(abs(v - pool_mean) for v in tab[a]['per_seed']), 4),
                             'zero_rollback_pass': bool(max(abs(v - pool_mean) for v in tab[a]['per_seed']) <= 0.02)}
                         for a in arms if a != 'pool'},
       'one_sided_non_inferior': {a: {'min_delta_vs_pool': round(min(v - pool_mean for v in tab[a]['per_seed']), 4),
                                      'pass_one_sided': bool(min(v - pool_mean for v in tab[a]['per_seed']) >= -0.02)}
                                  for a in arms if a != 'pool'},
       'reading': '双侧 |Δ|≤0.02 为 E16b 原判据（对「不劣」而言过严：高于 pool 亦被记 FAIL）；单侧不劣（min Δ ≥ −0.02）为实质读法，双报。',
       'lp_gain_blind_to_pool': round(pool_mean - tab['blind_learned']['mean'], 4),
       'lp_delta_heckman_vs_pool': round(tab['heckman']['mean'] - pool_mean, 4),
       'lp_delta_L0sel_vs_pool': round(tab['L0sel']['mean'] - pool_mean, 4),
       'missing_arms': ['patmix（E23 未实现；LANL 侧 E16b 有）'],
       'note': 'E16b 判据（|Δ|≤0.02 不劣）同构；本表为既有工件再表式化，不重跑'}
json.dump(rep, open(f'{OUT}/analysis_e28.json', 'w'), indent=1, ensure_ascii=False)
open(f'{OUT}/DONE.sentinel', 'w').close()
print(json.dumps(rep['zero_rollback'], ensure_ascii=False))
print(json.dumps({k: rep[k] for k in ('pool_mean', 'lp_gain_blind_to_pool', 'lp_delta_heckman_vs_pool', 'lp_delta_L0sel_vs_pool')}, ensure_ascii=False))
print('DONE E28')
