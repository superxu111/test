"""r2_e11_lanl_blind.py — E11：LANL 遮蔽三臂桥接实验（冷/热盲二分机理 × 真实企业数据闭环）。

使命：把 E1/E6/E8 在 enron10 上证明的「冷盲崩塌 / 热盲恢复」遮蔽三臂协议搬上 LANL cyber1
（E5 全观测主实验的数据与训练协议），检验机制发现在真实企业认证图上是否成立。

=== 臂定义与评估口径（先冻结后跑数；判据后置——LANL 无预注册判据，保持/变号均如实落盘） ===

臂（对齐 E1，训练协议逐位复用 E5 r1_e5_lanl_main.py）：
  full    全观测输入（E5 主实验同款：训练日 0-9 全量日图，EulerGCN，epochs≤300/patience 40/
          lr 5e-3/hidden 32/zdim 16/loss_sample 20000/Adam/clip 10/训练窗重构 val 选模——E5 审计
          已声明该 val 非 held-out，此处沿用保证臂间可比）
  blind0  冷盲：训练日 0 起全部 S 关联边从输入图移除（S 主机对模型全程不可见）
  blind5  热盲：训练日 0-4 全量、日 5 起遮蔽（模型对 S 有 5 天记忆前史）
  评分输入流（E5 DETECT 语义）：序列 = [训练窗 10 个(遮蔽)快照] + [评估日 d 的(遮蔽)快照]；
  blind 臂的评估日快照同样遮蔽（传感器自 onset 起失明）；被评分边集 = 该日全量唯一无向边。

遮蔽对象 S（攻击无关、确定性、仅用训练窗结构）：
  1) 主机全集 = E5 口径（C\d+ 过滤后的全部主机，N=17501）；
  2) 训练窗（日 0-9）聚合无向认证图；
  3) Louvain 社区（networkx，seed=7，resolution=1.0；enron10 用 greedy_modularity_communities，
     在 N=17.5k 上实测 >4min 不可行，Louvain 为同族模块度优化、固定种子确定性，如实替换披露）；
  4) 按社区规模降序整块吞入直至 |S| ≥ 0.4N（enron10 community_mask 同规则；LANL 社区粒度粗，
     实际达成率允许超 40%，落盘披露）。
  硬检查：评估事件跨界（S×V）数 ≥ 60，否则中止（跨界评估无意义）。

推理时混合（免训练，加在 blind 臂、仅跨界边；E6 pool 协议移植）：
  degfall 先验  a_deg = log1p(deg_causal[可见端])，deg_causal = 遮蔽流在日 <d 的累计度
                （E8 strict 因果口径：不含查询日；含前史全量日——对齐 E1 onset>0 的 deg_blind
                语义但去掉其未来日泄漏）。主口径累计到 d-1 的全部已过日；敏感性口径仅训练窗。
  pool          a_pool = λ*·rank(a_deg 方向按 σ_deg) + (1-λ*)·rank(a_learn)，rank 为
                (日,层) 内跨界边集合的秩归一（E1 ranknorm 语义，日块化）；λ*∈{0.3,0.5,0.7,0.9,1.0}、
                σ_deg∈{+1,-1}（先验通道符号，E1 σ 翻转机制移植到先验通道——检测方向下两通道
                方向均为经验问题）在验证日（训练窗内红队日 ∩ [onset,9]）跨界边上按检测 AP 选出。
  检测方向（E5 告警语义）：a_learn = -decode(u,v,z)（链路分低=异常，与 E5 raw/aci_tau 告警同向）。

评估（红队日 = E5 的 11 个评估日 12-29；标注 = 对在该日红队对中（E5 近似，E4 已证 98.1% 事件
精确 ts 命中 auth）；分层 = 跨界（恰一端∈S）vs 可见-可见（双端∉S）；S-S 边不进两主分层）：
  逐 (日,层)：AUC_det（层内攻击边 vs 非攻击边，事件加权 sample_weight=ev 与不加权双报）；
  汇总：日宏平均 AUC / 日块化池化 AUC（日内秩归一后合并）/ 攻击边中位低分百分位（E5
  median_pct 同向可比）；与 full 臂差值（配对 bootstrap 10000 次，日块化池化分，聚合器算）。
  双可见性口径：asimpl（z 含查询日快照，E5 原语义）/ strict（z=训练窗末态，E8 严格因果口径），
  两口径各自独立选 λ*/σ*、各自成表。
产物：results/r2/lanl_blindness/ 下 results_e11.jsonl（逐 arm×seed 全指标）+
scores_e11_{arm}_seed{}.npz（两口径逐日逐层分数与标签，免重训复分析）+ mask_S.json（遮蔽对象
清单+规则元数据）。与 full 臂配对差及总表在 r2_e11_analyze.py。
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/root/autodl-tmp/Euler-main')
sys.path.insert(0, '/root/autodl-tmp/Euler-main/benchmarks')
from models.euler_serial import EulerGCN  # noqa: E402

TRAIN_DAYS = list(range(0, 10))
EVAL_DAYS = [12, 13, 14, 15, 20, 21, 22, 26, 27, 28, 29]
VAL_RT_DAYS = [1, 2, 5, 6, 7, 8, 9]          # 训练窗内红队日（λ*/σ* 选择用）
LAM_GRID = (0.3, 0.5, 0.7, 0.9, 1.0)
SIG_GRID = (1, -1)
CONVS = ('asimpl', 'strict')


def ranknorm(v):
    v = np.asarray(v, dtype=float)
    order = v.argsort().argsort().astype(float)
    return order / max(len(v) - 1, 1)


def load_all():
    """E5 口径载入：日级边表 → 主机全集(N=17501) → 逐日唯一无向边编码 + 红队逐日事件。"""
    df = pd.read_csv('/root/autodl-tmp/data/processed/lanl_daily_edges.tsv', sep='\t',
                     header=None, names=['day', 'src', 'dst', 'nt', 'cnt'])
    ok = df.src.str.match(r'^C\d+$') & df.dst.str.match(r'^C\d+$')
    df = df[ok]
    hosts = sorted(set(df.src) | set(df.dst))
    hmap = {h: i for i, h in enumerate(hosts)}
    N = len(hosts)
    df['u'] = df.src.map(hmap)
    df['v'] = df.dst.map(hmap)
    df = df[df.u != df.v]
    # 逐日唯一无向边（编码 u*N+v，u<v）；与 E5 的 to_undirected 双向复制在分数上等价（decode 对称）
    day_pairs = {}
    for d, g in df.groupby('day'):
        u, v = g.u.values, g.v.values
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        day_pairs[int(d)] = np.unique(lo.astype(np.int64) * N + hi)
    ev_by_day_pair = {}
    with gzip.open('/root/autodl-tmp/data/raw/lanl/redteam.txt.gz', 'rt') as f:
        next(f)
        for line in f:
            ts, usr, s, t = line.strip().split(',')
            d = int(ts) // 86400
            if s not in hmap or t not in hmap:
                continue
            ev_by_day_pair.setdefault(d, {}).setdefault((hmap[s], hmap[t]), 0)
            ev_by_day_pair[d][(hmap[s], hmap[t])] += 1
    return day_pairs, ev_by_day_pair, N, hosts


def build_mask(day_pairs, N, out_dir, force=False):
    """冻结规则：训练窗聚合图 Louvain(seed=7) 社区按规模降序整块吞入至 |S|≥0.4N。"""
    path = os.path.join(out_dir, 'mask_S.json')
    if os.path.exists(path) and not force:
        j = json.load(open(path))
        return j['S'], j
    import networkx as nx
    keys = np.concatenate([day_pairs[d] for d in TRAIN_DAYS if d in day_pairs])
    us, vs = (keys // N).tolist(), (keys % N).tolist()
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(zip(us, vs))
    comms = sorted(nx.algorithms.community.louvain_communities(G, seed=7, resolution=1.0),
                   key=len, reverse=True)
    target = int(0.4 * N)
    S, sizes = set(), []
    for c in comms:
        if len(S) >= target:
            break
        S |= set(c)
        sizes.append(len(c))
    meta = {'rule': 'louvain(seed=7,res=1.0)_whole_comm_greedy_ge_0.4N',
            'N': int(N), 'target': target, 'achieved': len(S),
            'achieved_frac': round(len(S) / N, 4),
            'n_comms': len(comms), 'taken_comm_sizes': sizes,
            'top_comm_sizes': [len(c) for c in comms[:6]],
            'note': 'enron10 用 greedy_modularity_communities；N=17.5k 实测>4min 不可行，'
                    '换同族 Louvain（固定种子确定性），吞入规则其余逐位相同'}
    S = sorted(int(i) for i in S)
    json.dump({'S': S, **meta}, open(path, 'w'))
    return S, meta


def batch_loss(model, p, n, z, neg_weight=0.5):
    """E5 逐位复制的损失（EulerGCN.calc_loss 官方版固定 CPU，GPU 不可用）"""
    EPS = 1e-6
    pos = model.decode(p[0], p[1], z)
    neg = model.decode(n[0], n[1], z)
    return ((1 - neg_weight) * (-torch.log(pos + EPS)).mean()
            + neg_weight * (-torch.log(1 - neg + EPS)).mean())


def train_arm(model_input_days, N, dev, args, seed):
    """E5 训练循环逐位移植（输入换为各臂的日快照）。model_input_days: list[(u_arr, v_arr)]"""
    torch.manual_seed(seed)
    model = EulerGCN(N, h_dim=args.hidden, z_dim=args.zdim,
                     gru_hidden_units=1, use_w=True).to(dev)
    opt = Adam(model.parameters(), lr=args.lr)
    x = torch.eye(N, device=dev)
    ei_tr = [torch.tensor(np.vstack([u, v]), dtype=torch.long, device=dev)
             for u, v in model_input_days]
    T = len(ei_tr)
    rng = np.random.RandomState(seed)
    best, best_state, no_imp, ep_run = -1.0, None, 0, 0
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
        ep_run = ep + 1
        if avg > best:
            best, no_imp = avg, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience and ep > 50:
                break
        if ep % 50 == 0:
            print(f'  ep{ep} val_ap_auc={avg/2:.4f}', flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, best / 2, ep_run


def auc_pair(a, y, ev):
    """(日,层) 内检测 AUC：不加权 + 事件加权（正例按事件数加权，负例权 1）；
    无正例或无负例返回 None。"""
    out = {'auc': None, 'auc_eventw': None}
    if y.sum() >= 1 and (1 - y).sum() >= 1:
        out['auc'] = round(float(roc_auc_score(y, a)), 4)
        out['auc_eventw'] = round(float(
            roc_auc_score(y, a, sample_weight=np.where(y == 1, ev, 1.0))), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['full', 'blind0', 'blind5'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--zdim', type=int, default=16)
    ap.add_argument('--loss-sample', type=int, default=20000)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='/root/autodl-tmp/results/r2/lanl_blindness')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    onset = {'full': 10 ** 9, 'blind0': 0, 'blind5': 5}[args.arm]  # full=永不遮蔽
    torch.manual_seed(2026)
    np.random.seed(2026)
    dev = torch.device(args.device)

    print(f'[E11 {args.arm} seed{args.seed}{args.tag}] loading data...', flush=True)
    day_pairs, ev_by_day_pair, N, hosts = load_all()
    S, mask_meta = build_mask(day_pairs, N, args.out)
    Smask = np.zeros(N, dtype=bool)
    Smask[np.asarray(S)] = True
    print(f"N={N} |S|={len(S)} ({len(S)/N:.3f}) taken_comm_sizes={mask_meta['taken_comm_sizes']}",
          flush=True)

    # ---- 红队逐日事件与分层覆盖（硬检查：跨界事件>=60）；标注覆盖评估日+验证日 ----
    rt_by_day = {}
    cov = {'cross': 0, 'vv': 0, 'ss': 0}
    for d in sorted(set(EVAL_DAYS) | set(VAL_RT_DAYS)):
        mp = ev_by_day_pair.get(d, {})
        marks = {}
        for (a, b), ev in mp.items():
            marks[(min(a, b), max(a, b))] = marks.get((min(a, b), max(a, b)), 0) + ev
            if d in EVAL_DAYS:
                k = 'ss' if (Smask[a] and Smask[b]) else (
                    'vv' if not (Smask[a] or Smask[b]) else 'cross')
                cov[k] += ev
        rt_by_day[d] = marks
    print(f'eval events: cross={cov["cross"]} vv={cov["vv"]} ss={cov["ss"]} '
          f'(cross_frac={cov["cross"]/max(sum(cov.values()),1):.3f})', flush=True)
    assert cov['cross'] >= 60, f'cross coverage {cov["cross"]} < 60 — 遮蔽对象选取失败，中止'

    # ---- 各臂遮蔽流日快照 + 因果累计度快照（<d 口径） ----
    def masked_keys(day):
        keys = day_pairs.get(day, np.array([], dtype=np.int64))
        u, v = keys // N, keys % N
        if args.arm == 'full' or day < onset:
            keep = np.ones(len(keys), dtype=bool)
        else:
            keep = ~(Smask[u] | Smask[v])
        return u[keep], v[keep]

    train_days_input = [masked_keys(d) for d in TRAIN_DAYS]
    need_days = sorted(set(EVAL_DAYS) | set(VAL_RT_DAYS))
    deg_snap_allpast, deg_snap_trwin = {}, {}
    deg_all = np.zeros(N, dtype=np.int64)   # 遮蔽流全部已过日累计
    deg_win = np.zeros(N, dtype=np.int64)   # 遮蔽流训练窗(0-9)累计
    for d in range(0, max(need_days) + 1):
        if d in need_days:                   # 快照 = 日 d 之前(<d)的累计度
            deg_snap_allpast[d] = deg_all.copy()
            deg_snap_trwin[d] = deg_win.copy()
        u, v = masked_keys(d)
        np.add.at(deg_all, u, 1)
        np.add.at(deg_all, v, 1)
        if d <= 9:
            np.add.at(deg_win, u, 1)
            np.add.at(deg_win, v, 1)

    # ---- 训练 ----
    print(f'training arm={args.arm} seed={args.seed} (onset={args.arm})...', flush=True)
    t0 = time.time()
    model, best_val, ep_run = train_arm(train_days_input, N, dev, args, args.seed)
    print(f'training done: best_val={best_val:.4f} epochs_run={ep_run} '
          f'({time.time()-t0:.0f}s) [E5 seed0 参考 0.9951]', flush=True)
    x = torch.eye(N, device=dev)
    ei_tr = [torch.tensor(np.vstack([u, v]), dtype=torch.long, device=dev)
             for u, v in train_days_input]

    # ---- 逐日打分：验证日（训练序列内）+ 评估日（训练窗+当日序列）；双口径 ----
    # 注：序列中的当日快照=遮蔽流（输入端失明）；decode 目标=该日全量唯一边
    # （跨界边即使不在输入图中也被打分——遮蔽协议的评估语义）
    with torch.no_grad():
        zs_tr = model(x, ei_tr, lambda i: ei_tr[i])   # (T,N,z)
    per_day = {}
    for d in sorted(set(EVAL_DAYS) | set(VAL_RT_DAYS)):
        day_keys = day_pairs.get(d, np.array([], dtype=np.int64))
        u_all, v_all = day_keys // N, day_keys % N
        mu, mv = masked_keys(d)
        masked_ei = torch.tensor(np.vstack([mu, mv]), dtype=torch.long, device=dev)
        full_ei = torch.tensor(np.vstack([u_all, v_all]), dtype=torch.long, device=dev)
        if d in VAL_RT_DAYS and d in TRAIN_DAYS:
            with torch.no_grad():
                sc_asimpl = model.decode(full_ei[0], full_ei[1], zs_tr[d]).cpu().numpy()
                sc_strict = (model.decode(full_ei[0], full_ei[1], zs_tr[d - 1]).cpu().numpy()
                             if d >= 1 else sc_asimpl)
        else:
            seq = ei_tr + [masked_ei]
            with torch.no_grad():
                zs = model(x, seq, lambda i: seq[i])
                sc_asimpl = model.decode(full_ei[0], full_ei[1], zs[-1]).cpu().numpy()
                sc_strict = model.decode(full_ei[0], full_ei[1],
                                         zs[len(ei_tr) - 1]).cpu().numpy()
        per_day[d] = {'u': u_all, 'v': v_all,
                      'a_asimpl': -sc_asimpl.astype(np.float64),   # 检测方向 a=-decode
                      'a_strict': -sc_strict.astype(np.float64)}

    # ---- (日,层) 数组缓存 ----
    def day_stratum_arrays(d, stratum):
        pd_ = per_day[d]
        su, sv = Smask[pd_['u']], Smask[pd_['v']]
        m = (su ^ sv) if stratum == 'cross' else ~(su | sv)
        u, v = pd_['u'][m], pd_['v'][m]
        mp = rt_by_day.get(d, {})
        ev = np.array([mp.get((min(int(a), int(b)), max(int(a), int(b))), 0)
                       for a, b in zip(u, v)], dtype=np.int64)
        vis = np.where(Smask[u], v, u)
        return {'u': u.astype(np.int32), 'v': v.astype(np.int32),
                'y': (ev > 0).astype(np.int8), 'ev': ev,
                'a_asimpl': pd_['a_asimpl'][m], 'a_strict': pd_['a_strict'][m],
                'deg_allpast': deg_snap_allpast[d][vis],
                'deg_trwin': deg_snap_trwin[d][vis]}

    ARR = {(d, s): day_stratum_arrays(d, s)
           for d in sorted(set(EVAL_DAYS) | set(VAL_RT_DAYS)) for s in ('cross', 'vv')}

    # ---- 验证日 λ*/σ* 选择（仅 blind 臂实际使用；full 臂也算出存档不使用） ----
    lam_sig = {}
    val_days_arm = [d for d in VAL_RT_DAYS if d >= onset]
    for conv in CONVS:
        vp_a, vn_a, vp_d, vn_d = [], [], [], []
        for d in val_days_arm:
            arr = ARR[(d, 'cross')]
            if arr['y'].sum() == 0:
                continue
            vp_a.append(arr[f'a_{conv}'][arr['y'] == 1])
            vp_d.append(np.log1p(arr['deg_allpast'][arr['y'] == 1]))
            vn_a.append(arr[f'a_{conv}'][arr['y'] == 0])
            vn_d.append(np.log1p(arr['deg_allpast'][arr['y'] == 0]))
        if not vp_a:
            lam_sig[conv] = {'lam_star': None, 'sig_deg': None, 'va_ap_table': {},
                             'n_va_pos': 0, 'val_days_used': val_days_arm}
            continue
        va = np.concatenate(vp_a + vn_a)
        vd = np.concatenate(vp_d + vn_d)
        vlab = np.concatenate([np.ones(sum(len(z) for z in vp_a)),
                               np.zeros(sum(len(z) for z in vn_a))])
        ra, rd = ranknorm(va), ranknorm(vd)
        table = {}
        for sig in SIG_GRID:
            for lam in LAM_GRID:
                s = lam * (rd if sig == 1 else 1 - rd) + (1 - lam) * ra
                table[f'sig{sig}_lam{lam}'] = round(float(
                    average_precision_score(vlab, s)), 4)
        bk = max(table, key=table.get)
        lam_sig[conv] = {'lam_star': float(bk.split('_lam')[1]),
                         'sig_deg': int(bk.split('_')[0][3:]),
                         'va_ap_table': table, 'n_va_pos': int(vlab.sum()),
                         'val_days_used': val_days_arm}

    # ---- 断点续跑 ----
    outp = os.path.join(args.out, 'results_e11.jsonl')
    if os.path.exists(outp):
        for line in open(outp):
            try:
                r_ = json.loads(line)
                if r_['arm'] == args.arm and r_['seed'] == args.seed \
                        and r_.get('tag', '') == args.tag:
                    print(f'resume: {args.arm} seed{args.seed}{args.tag} already done — skip',
                          flush=True)
                    return
            except Exception:
                pass

    # ---- 评估指标 + npz ----
    metrics = {c: {} for c in CONVS}
    npz = {}
    for d in sorted(set(EVAL_DAYS) | set(VAL_RT_DAYS)):
        for s in ('cross', 'vv'):
            a = ARR[(d, s)]
            base = {f'{d}_{s}_u': a['u'], f'{d}_{s}_v': a['v'],
                    f'{d}_{s}_y': a['y'], f'{d}_{s}_ev': a['ev'],
                    f'{d}_{s}_deg': a['deg_allpast'].astype(np.int64),
                    f'{d}_{s}_degtr': a['deg_trwin'].astype(np.int64)}
            for c in CONVS:
                base[f'{c}__{d}_{s}_a'] = a[f'a_{c}'].astype(np.float32)
            npz.update(base)
    npz.update({f'lam_sig_{c}': json.dumps(lam_sig[c]) for c in CONVS})

    for conv in CONVS:
        for stratum in ('cross', 'vv'):
            rows = []
            pooled_r, pooled_y, pooled_w = [], [], []
            pooled_rd, pooled_rp, pooled_rpt = [], [], []
            all_att_pct = []
            for d in EVAL_DAYS:
                arr = ARR[(d, stratum)]
                if len(arr['y']) == 0:
                    continue
                a = arr[f'a_{conv}']
                y, ev = arr['y'], arr['ev']
                r = ranknorm(a)                     # (日,层) 内检测秩
                res = auc_pair(a, y, ev)
                row = {'day': d, 'n_edges': int(len(y)),
                       'n_attack_edges': int(y.sum()),
                       'n_attack_events': int(ev[y == 1].sum()) if y.sum() else 0,
                       'auc': res['auc'], 'auc_eventw': res['auc_eventw']}
                if y.sum():
                    row['median_pct_low'] = round(float(np.median(1.0 - r[y == 1])), 4)
                    all_att_pct.append(1.0 - r[y == 1])
                if stratum == 'cross' and args.arm != 'full':
                    ls = lam_sig[conv]
                    assert ls['lam_star'] is not None, '盲臂验证日无跨界正例，λ* 未选出'
                    ld = np.log1p(arr['deg_allpast'])
                    ld_tr = np.log1p(arr['deg_trwin'])
                    a_deg = ld if ls['sig_deg'] == 1 else -ld
                    rd_ = ranknorm(ld if ls['sig_deg'] == 1 else -ld)
                    rd_tr = ranknorm(ld_tr if ls['sig_deg'] == 1 else -ld_tr)
                    a_pool = ls['lam_star'] * rd_ + (1 - ls['lam_star']) * r
                    a_pool_tr = ls['lam_star'] * rd_tr + (1 - ls['lam_star']) * r
                    m_deg = auc_pair(a_deg, y, ev)
                    m_pool = auc_pair(a_pool, y, ev)
                    m_pool_tr = auc_pair(a_pool_tr, y, ev)
                    row.update({'auc_degfall': m_deg['auc'],
                                'auc_degfall_eventw': m_deg['auc_eventw'],
                                'auc_pool': m_pool['auc'],
                                'auc_pool_eventw': m_pool['auc_eventw'],
                                'auc_pool_trwin': m_pool_tr['auc']})
                    pooled_rd.append(rd_)
                    pooled_rp.append(ranknorm(a_pool))
                    pooled_rpt.append(ranknorm(a_pool_tr))
                    npz[f'{conv}__{d}_{stratum}_deg'] = a_deg.astype(np.float32)
                    npz[f'{conv}__{d}_{stratum}_pool'] = a_pool.astype(np.float32)
                rows.append(row)
                pooled_r.append(r); pooled_y.append(y); pooled_w.append(ev)
            pr = np.concatenate(pooled_r); py = np.concatenate(pooled_y)
            pw = np.where(py == 1, np.concatenate(pooled_w), 1.0)  # 正例事件加权，负例权 1
            agg = {'days': rows, 'n_edges': int(len(py)),
                   'n_attack_edges': int(py.sum()),
                   'n_attack_events': int(pw[py == 1].sum()) if py.sum() else 0}
            if py.sum() and (1 - py).sum():
                agg['pooled_auc_dayblocked'] = round(float(roc_auc_score(py, pr)), 4)
                agg['pooled_auc_dayblocked_eventw'] = round(
                    float(roc_auc_score(py, pr, sample_weight=pw)), 4)
                day_aucs = [q['auc'] for q in rows if q['auc'] is not None]
                day_aucw = [q['auc_eventw'] for q in rows if q['auc_eventw'] is not None]
                agg['macro_auc'] = round(float(np.mean(day_aucs)), 4) if day_aucs else None
                agg['macro_auc_eventw'] = (round(float(np.mean(day_aucw)), 4)
                                           if day_aucw else None)
                agg['median_pct_low_pooled'] = round(
                    float(np.median(np.concatenate(all_att_pct))), 4) if all_att_pct else None
            if stratum == 'cross' and args.arm != 'full' and pooled_rp:
                agg['pooled_auc_degfall'] = round(float(roc_auc_score(
                    py, np.concatenate(pooled_rd))), 4)
                agg['pooled_auc_pool'] = round(float(roc_auc_score(
                    py, np.concatenate(pooled_rp))), 4)
                agg['pooled_auc_pool_trwin'] = round(float(roc_auc_score(
                    py, np.concatenate(pooled_rpt))), 4)
            metrics[conv][stratum] = agg

    rec = {'arm': args.arm, 'seed': args.seed, 'tag': args.tag,
           'best_val': round(best_val, 4), 'epochs_run': ep_run,
           'mask': {k: mask_meta[k] for k in ('achieved_frac', 'taken_comm_sizes', 'rule')},
           'eval_event_coverage': cov, 'lam_sig': lam_sig, 'metrics': metrics,
           'sec': round(time.time() - t_start, 1)}

    with open(outp, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    np.savez_compressed(os.path.join(
        args.out, f'scores_e11_{args.arm}_seed{args.seed}{args.tag}.npz'), **npz)
    mc = metrics['asimpl']['cross']
    mv = metrics['asimpl']['vv']
    ls_a = lam_sig['asimpl']
    print(f"[{args.arm} seed{args.seed}] CROSS asimpl pooled={mc.get('pooled_auc_dayblocked')} "
          f"pool={mc.get('pooled_auc_pool')} degfall={mc.get('pooled_auc_degfall')} "
          f"lam*={ls_a['lam_star']} sig={ls_a['sig_deg']} | "
          f"VV pooled={mv.get('pooled_auc_dayblocked')} ({rec['sec']}s)", flush=True)
    print('E11_RUN_DONE', args.arm, args.seed, flush=True)


if __name__ == '__main__':
    main()
