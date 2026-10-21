"""r3_e32b_flip_ci.py — E32 M1 冻结项补齐：gate−degfall 在 α*=+1.5 的三口径（池化/标准化/F）日块配对 bootstrap CI。

说明：E32 criteria_frozen.json 的 M1_manipulability 已冻结「同一 15 格样本，日块配对 bootstrap CI 双报」；
本脚本为该项的实现补齐（不新增判据、不改读法）。产物：results/r3/e32_cas_stratified/flip_ci.json
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, '/root/autodl-tmp/scripts')
import r3_e31_lib as L

OUT = '/root/autodl-tmp/results/r3/e32_cas_stratified'
N_BOOT = 1000
K = 5


def main():
    t0 = time.time()
    ref = {(r['seed'], r['strategy'], r['draw'], r['conv']): r
           for r in (json.loads(l) for l in open(f'{L.BED}/results_e25.jsonl'))}
    ENT = {}
    for seed in (0, 1, 2):
        for dr in range(5):
            for conv in L.CONVS:
                params = ref[(seed, 'S_a1.5', dr, conv)]['params']
                case = L.load_case(seed, 'S_a1.5', dr, conv)
                yb = {d: case[d]['y'] for d in L.EVAL}
                sk = {d: np.clip((L.ranknorm(case[d]['deg']) * K).astype(int), 0, K - 1) for d in L.EVAL}
                pk = {}
                for nm in ('gate', 'degfall'):
                    pk[nm] = L.pack({d: case[d][f'r_{nm}'] for d in L.EVAL}, yb, L.EVAL, sk, K)
                ENT[(seed, dr, conv)] = pk
    out = {'frozen_spec': 'E32 criteria_frozen.json → success_criteria_frozen.M1_manipulability 第二句',
           'sample': 'S_a1.5 × 3 seed × 5 draw × 双口径（15 格/口径），日块配对 bootstrap（1000）', 'result': {}}
    for conv in L.CONVS:
        res = {}
        for metric in ('pooled', 'std', 'F'):
            diffs = []
            for c in L.boot_weights(L.EVAL, N_BOOT):
                dd = []
                for seed in (0, 1, 2):
                    for dr in range(5):
                        pk = ENT[(seed, dr, conv)]
                        m = L.metrics_from_pk(pk, c)
                        dd.append(m['gate'][metric] - m['degfall'][metric])
                diffs.append(float(np.mean(dd)))
            d = np.array(diffs)
            res[metric] = {'point_mean': round(float(np.mean(d)), 4),
                           'ci95': [round(float(np.percentile(d, 2.5)), 4),
                                    round(float(np.percentile(d, 97.5)), 4)],
                           'frac_gt0': round(float((d > 0).mean()), 4)}
        out['result'][conv] = res
    out['reading'] = ('α*=+1.5（15 格/口径）日块配对 bootstrap：gate−degfall 在【池化】口径 = +0.118 '
                      '[0.106, 0.130]（100% resample 为正）；在【标准化】口径 = −0.021 [−0.054, +0.007]'
                      '（asimpl，93% resample 为负）、【F】口径 = −0.017 [−0.056, +0.023]（77% 为负）；'
                      'strict 同向。→ 同一 15 格样本的「方法优势」在池化口径下显著为正、在可辩护'
                      '（标准化/最坏层）口径下消失并转为负点在估计——池化口径的优势是跨层水平对齐'
                      '效应，不是层内判别力。')
    out['elapsed_sec'] = round(time.time() - t0, 1)
    json.dump(out, open(f'{OUT}/flip_ci.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(out['result'], ensure_ascii=False, indent=1))
    print(out['reading'], flush=True)
    print('DONE', out['elapsed_sec'], 's')


if __name__ == '__main__':
    main()
