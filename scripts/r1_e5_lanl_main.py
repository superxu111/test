"""r1_e5_lanl_main.py — R1实验E5：LANL auth 主实验（Q16，最小可发表版）。
管线：auth 日级聚合边表（E5 prep 产物，NTLM 行=官方 Euler 过滤协议）→ EulerGCN 日级序列 →
逐攻击日全边评分 → 预算-召回曲线（K∈{5,10,20,50}）+ 攻击日分层（安静 vs top）+ oracle 相对召回。
协议要点：
  - 训练天 = 绝对日 0..9（无监督打分器；红队事件占比 ~4e-6，污染可忽略并在论文声明）；
    评估天 = 训练窗之后的红队日 {12,13,14,15,20,21,22,26,27,28,29}（11 天，411 事件）
  - 校准 = 评估天前 7 天逐日 kth 分数池（在线滚动；窗口含红队日，无干净期可用，如实报告）
  - 命中语义两级：pair 级（告警边=(src,dst)∈当日红队对）与 host 级（告警边触及当日活跃攻击主机）；
    E4 取证裁决 pair 级结构性 0 后以 host 级+oracle 相对召回为主报
  - oracle：pair 级 = Σ_d min(events_d,K)/Σ events_d（R2 的 0.291 同型）；host 级 = 当日攻击主机
    关联边中可被 K 名额覆盖的比例
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/root/autodl-tmp/Euler-main')
sys.path.insert(0, '/root/autodl-tmp/Euler-main/benchmarks')
from models.euler_serial import EulerGCN  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402

TRAIN_DAYS = list(range(0, 10))
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
CAL_WIN = 7


def lg(s):
    s = np.clip(s, 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def batch_loss(model, p, n, z, neg_weight=0.5):
    """EulerGCN.calc_loss 的等价实现（官方 loss_fn 的 zeros((1)) 固定 CPU，GPU 下不可用）"""
    EPS = 1e-6
    pos = model.decode(p[0], p[1], z)
    neg = model.decode(n[0], n[1], z)
    return ((1 - neg_weight) * (-torch.log(pos + EPS)).mean()
            + neg_weight * (-torch.log(1 - neg + EPS)).mean())


def wquantile(vals, q):
    s = np.sort(np.asarray(vals, float))
    D = len(s)
    i = int(np.searchsorted(np.cumsum(np.ones(D) / D), q))
    return float(s[min(i, D - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', default='/root/autodl-tmp/data/processed/lanl_daily_edges.tsv')
    ap.add_argument('--redteam', default='/root/autodl-tmp/data/raw/lanl/redteam.txt.gz')
    ap.add_argument('--ntlm', action='store_true',
                    help='仅用 NTLM 行（官方 Euler 过滤协议）；默认用全部 C* 主机对（147k 对/日）')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--zdim', type=int, default=16)
    ap.add_argument('--loss-sample', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='/root/autodl-tmp/results/r1/e5_lanl')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    dev = torch.device(args.device)

    # ---------- 载入日级边表（NTLM 行） ----------
    t0 = time.time()
    df = pd.read_csv(args.edges, sep='\t', header=None,
                     names=['day', 'src', 'dst', 'nt', 'cnt'])
    print(f'edges rows={len(df)} load {time.time()-t0:.0f}s', flush=True)
    # 主机名过滤（LANL 短格式行污染 ~6.5% 行/1.2% 事件，C\d+ 为官方主机命名）
    ok = df.src.str.match(r'^C\d+$') & df.dst.str.match(r'^C\d+$')
    df = df[ok]
    if args.ntlm:
        df = df[df.nt == 1]
    hosts = sorted(set(df.src) | set(df.dst))
    hmap = {h: i for i, h in enumerate(hosts)}
    N = len(hosts)
    print(f'clean rows={len(df)} hosts N={N}', flush=True)
    df['u'] = df.src.map(hmap)
    df['v'] = df.dst.map(hmap)
    df = df[df.u != df.v]

    eis = {}
    day_events = {}
    for d, g in df.groupby('day'):
        ei = to_undirected(torch.tensor(
            np.vstack([g.u.values, g.v.values]), dtype=torch.long))
        eis[int(d)] = ei
        day_events[int(d)] = int(g.cnt.sum())
    alldays = sorted(eis)
    print(f'days={alldays[0]}..{alldays[-1]} ({len(alldays)}), '
          f'daily uniq edges med={int(np.median([eis[d].size(1) for d in alldays]))}', flush=True)

    # ---------- 红队真值（E4b 事件级标注：734/748 事件精确 ts 命中 auth；按日标注攻击对） ----------
    import gzip
    join = json.load(open('/root/autodl-tmp/results/r1/e4_forensics/rt_ts_join.json'))
    # 每日攻击对 -> 事件数（用于事件加权召回）
    ev_by_day_pair = {}
    with gzip.open(args.redteam, 'rt') as f:
        next(f)
        for line in f:
            ts, usr, s, t = line.strip().split(',')
            d = int(ts) // 86400
            key = (hmap.get(s, -1), hmap.get(t, -1))
            if key[0] < 0 or key[1] < 0:
                continue
            ev_by_day_pair.setdefault(d, {}).setdefault(key, 0)
            # 只统计精确-ts 命中 auth 的事件（Euler 语义）；近似：全部事件（98.1% 命中）
            ev_by_day_pair[d][key] += 1
    rt_pairs = {d: set(mp) for d, mp in ev_by_day_pair.items()}
    rt_events = {d: sum(mp.values()) for d, mp in ev_by_day_pair.items()}
    rt_hosts = {}
    for d, mp in ev_by_day_pair.items():
        h = set()
        for (a, b) in mp:
            h.add(a); h.add(b)
        rt_hosts[d] = h
    print(f'attack days={sorted(rt_events)} marked_pairs/day='
          f'{({d: len(p) for d, p in sorted(rt_pairs.items())})}', flush=True)

    # ---------- 训练（日级 EulerGCN，GPU） ----------
    tr_days = [d for d in TRAIN_DAYS if d in eis]
    ei_tr = [eis[d].to(dev) for d in tr_days]
    x = torch.eye(N, device=dev)
    torch.manual_seed(args.seed)
    model = EulerGCN(N, h_dim=args.hidden, z_dim=args.zdim,
                     gru_hidden_units=1, use_w=True).to(dev)
    opt = Adam(model.parameters(), lr=args.lr)
    T = len(ei_tr)
    end_tr = T  # 训练窗全部快照
    rng = np.random.RandomState(args.seed)
    best, best_state, no_imp = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        zs = model(x, ei_tr, lambda i: ei_tr[i])
        tl = 0.0
        for i in range(T):
            ei = ei_tr[i]
            ne = ei.size(1)
            k = min(args.loss_sample, ne)
            idx = torch.from_numpy(rng.randint(0, ne, k)).to(dev)
            pe = ei[:, idx]
            nu, nv = rng.randint(0, N, k), rng.randint(0, N, k)
            bad = nu == nv
            nu[bad] = (nu[bad] + 1) % N
            ne_ = torch.tensor(np.vstack([nu, nv]), dtype=torch.long, device=dev)
            tl = tl + batch_loss(model, (pe[0], pe[1]), (ne_[0], ne_[1]), zs[i])
        tl = tl / T
        tl.backward()
        opt.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        with torch.no_grad():
            model.eval()
            zs = model(x, ei_tr, lambda i: ei_tr[i])
            st, sf = [], []
            for i in range(T):
                ei = ei_tr[i]
                k = min(5000, ei.size(1))
                idx = torch.from_numpy(rng.randint(0, ei.size(1), k)).to(dev)
                pe = ei[:, idx]
                nu, nv = rng.randint(0, N, k), rng.randint(0, N, k)
                bad = nu == nv
                nu[bad] = (nu[bad] + 1) % N
                ne_ = torch.tensor(np.vstack([nu, nv]), dtype=torch.long, device=dev)
                st.append(model.decode(pe[0], pe[1], zs[i]))
                sf.append(model.decode(ne_[0], ne_[1], zs[i]))
            st = torch.cat(st); sf = torch.cat(sf)
            lab = torch.cat([torch.ones_like(st), torch.zeros_like(sf)])
            sc = torch.cat([st, sf])
            avg = (average_precision_score(lab.cpu().numpy(), sc.cpu().numpy())
                   + roc_auc_score(lab.cpu().numpy(), sc.cpu().numpy()))
        if avg > best:
            best, no_imp = avg, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience and ep > 50:
                break
        if ep % 50 == 0:
            print(f'ep{ep} val_ap_auc={avg/2:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    print(f'training done: best_val={best/2:.4f}', flush=True)

    # ---------- 评分流：评估日 + 其前 CAL_WIN 校准窗（DETECT 语义=序列末含当日图） ----------
    score_days = sorted(set(range(min(EVAL_DAYS) - CAL_WIN, max(EVAL_DAYS) + 1)))
    streams = {}
    for d in score_days:
        seq = ei_tr + [eis[d].to(dev)]
        with torch.no_grad():
            zs = model(x, seq, lambda i: seq[i])
            ei = eis[d]
            sc = model.decode(ei[0], ei[1], zs[-1]).cpu().numpy()
        streams[d] = sc
        print(f'scored day{d}: n_edges={len(sc)}', flush=True)

    # ---------- 预算机制与指标 ----------
    out = {'K': {}, 'protocol': {'train_days': tr_days, 'eval_days': EVAL_DAYS,
                                 'cal_win': CAL_WIN, 'n_hosts': int(N),
                                 'day_events': {str(d): day_events.get(d, 0) for d in EVAL_DAYS}}}
    eval_days = [d for d in EVAL_DAYS if d in rt_events]
    total_ev = sum(rt_events[d] for d in eval_days)

    # ---- E5-v2：事件级预算-召回扫 + 攻击对排名分位数（排名机制=分数升序，越靠前越异常）----
    KS = (5, 10, 20, 50, 100, 500, 1000)
    sweep = {K: {'hit_events': 0, 'hit_days': 0} for K in KS}
    pct_samples = []
    day_rank_stats = []
    for d in eval_days:
        sc = streams[d]
        order = np.argsort(sc, kind='stable')          # 升序：最低分（最异常）在前
        rank_of = np.empty(len(sc), dtype=np.int64)
        rank_of[order] = np.arange(len(sc))
        ei_np = eis[d].cpu().numpy()
        mp = rt_pairs[d]
        mpe = ev_by_day_pair[d]   # 对 -> 事件数
        n_mark = len(mp)
        marked_idx, marked_ev = [], []
        for i in range(ei_np.shape[1]):
            for key in ((int(ei_np[0][i]), int(ei_np[1][i])),
                        (int(ei_np[1][i]), int(ei_np[0][i]))):
                if key in mp:
                    marked_idx.append(i)
                    marked_ev.append(mpe[key])
        if not marked_idx:
            day_rank_stats.append({'day': int(d), 'n_marked': 0})
            continue
        ranks = rank_of[np.array(marked_idx)]
        pct = ranks / max(len(sc) - 1, 1)
        pct_samples.extend(pct.tolist())
        evs = np.array(marked_ev, dtype=float)
        for K in KS:
            hit = ranks < K
            sweep[K]['hit_events'] += float(evs[hit].sum())
            sweep[K]['hit_days'] += int(hit.any())
        day_rank_stats.append({'day': int(d), 'n_marked': int(n_mark),
                               'n_marked_edges': int(len(marked_idx)),
                               'median_pct': float(np.median(pct)),
                               'top100_hits': int((ranks < 100).sum())})
    for K in KS:
        sweep[K]['recall_vs_events'] = round(sweep[K]['hit_events'] / total_ev, 4)
        sweep[K]['oracle'] = round(sum(min(rt_events[d], K) for d in eval_days) / total_ev, 4)
    out2 = {'eval_events': int(total_ev), 'sweep': sweep,
            'median_rank_pct_pooled': round(float(np.median(pct_samples)), 4) if pct_samples else None,
            'days': day_rank_stats}
    with open(os.path.join(args.out, 'results_e5_eventwise.json'), 'w') as fh:
        json.dump(out2, fh, indent=1)
    print('EVENTWISE:', json.dumps({'sweep': {str(k): v for k, v in sweep.items()},
                                    'median_pct': out2['median_rank_pct_pooled']}), flush=True)

    for K in (5, 10, 20, 50):
        pair_oracle = sum(min(rt_events[d], K) for d in eval_days) / total_ev
        # 逐日递推
        pool = []           # 滚动校准池（kth scores, logit）
        tau = None
        cum_e = 0.0
        rows = []
        for d in sorted(eval_days):
            # 校准池：评估日之前 CAL_WIN 天的当日 kth（logit）
            cal = [lg(np.sort(streams[dd])[:K][-1])
                   for dd in score_days if dd < d][-CAL_WIN:]
            tau_raw = wquantile(cal, 0.9)
            if tau is None:
                tau = tau_raw
            # 在线反馈（aci_tau，logit 空间，预注册 η=0.1, 增益 3×std）
            ss = float(np.std(cal) + 1e-9)
            sc_lg = lg(streams[d])
            alarms_raw = int((sc_lg <= tau_raw).sum())
            alarms_aci = int((sc_lg <= tau).sum())
            kth_q = float(np.sort(sc_lg)[:K][-1])
            # host 级命中
            act = rt_hosts[d]
            ei_np = eis[d].cpu().numpy()
            is_attack_edge = np.isin(ei_np[0], list(act)) | np.isin(ei_np[1], list(act))
            f = {}
            for name, al in (('raw', alarms_raw), ('aci_tau', alarms_aci), ('quota', K)):
                # 告警边 = 最低 al 分的边（<=τ 等价于分数升序前 al 个——同序）
                order = np.argsort(sc_lg)
                sel = order[:al]
                pair_hit = int(sum(1 for i in sel
                                   if (int(ei_np[0][i]), int(ei_np[1][i])) in rt_pairs[d]
                                   or (int(ei_np[1][i]), int(ei_np[0][i])) in rt_pairs[d]))
                host_hit = int(is_attack_edge[sel].sum())
                f[name] = {'alarms': int(al), 'pair_hit': pair_hit, 'host_hit': host_hit}
            rows.append({'day': int(d), 'events': rt_events[d],
                         'n_obs': int(len(sc_lg)), 'n_attack_edges_obs': int(is_attack_edge.sum()),
                         **f})
            # 更新在线状态
            e = (alarms_aci - K) / K
            tau = tau - 0.1 * ss * e * 3.0
            cum_e += e
        ev = sum(r['events'] for r in rows)
        host_oracle = sum(min(K, r['n_attack_edges_obs']) for r in rows)
        host_obs = sum(r['n_attack_edges_obs'] for r in rows)
        res = {'pair_oracle': round(float(pair_oracle), 4), 'eval_events': int(ev), 'days': rows,
               'host_oracle_capture': int(host_oracle), 'host_attack_edges_total': int(host_obs)}
        for name in ('raw', 'aci_tau', 'quota'):
            ok = np.mean([r[name]['alarms'] <= K for r in rows])
            pair_rec = sum(r[name]['pair_hit'] for r in rows) / ev
            host_hit = sum(r[name]['host_hit'] for r in rows)
            res[name] = {'budget_ok': round(float(ok), 3),
                         'pair_recall': round(float(pair_rec), 4),
                         'host_hit_total': int(host_hit),
                         'host_capture_frac': (round(host_hit / host_oracle, 4)
                                               if host_oracle else None)}
        out['K'][str(K)] = res
        print(f"K={K} pair_oracle={pair_oracle:.3f} " +
              " ".join(f"{n}: ok={res[n]['budget_ok']} pair_rec={res[n]['pair_recall']} "
                       f"host_cap={res[n]['host_capture_frac']}"
                       for n in ('raw', 'aci_tau', 'quota')), flush=True)
    with open(os.path.join(args.out, 'results_e5.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('E5_DONE', flush=True)


if __name__ == '__main__':
    main()
