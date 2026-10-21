"""r1_e1_module1.py — R1 闸门实验E1：模块①观测条件化评分·mask位置消融（预注册）。
背景（R2 smoke3 裁决）：blind 0.23-0.31 崩塌 / maskcond(input通道)≈blind 零修复 / degfall 0.76-0.81 救回65-82%。
本轮问题（toVerify Q1/Q2/Q3/Q5）：
  Q1 新 maskcond（mask进GRU更新门 或 读出评分混合）能否把跨界边AUC拉到≥0.7 且与full差≤0.15（3种子）
  Q2 三位置对照：input(R2已证零修复,复用) vs gate(记忆更新门) vs pool(读出打分)——修复来自哪个位置
  Q3 学习版能否显著超过 degfall 0.76-0.81（配对bootstrap）
  Q5 第五臂 opinion pool：score=λ·rank(degfall)+(1−λ)·rank(learned)，盲端点λ大、可见端λ=0——
     能否吃掉 degfall(0.76-0.81)→full(0.95-0.96) 残差空隙
臂：
  full        全图输入（对照上界）
  blind       盲化输入（崩塌基线）
  input       盲化+mask特征通道（R2最小mask条件化，预期零修复，复跑作同评估集对照）
  gate_frz    盲化+mask进GRU更新门（盲节点记忆冻结 h_t=h_{t-1}）
  gate_dec    盲化+mask进GRU更新门（盲节点记忆指数衰减 exp(-softplus δ)·h_{t-1}，δ可学习）
  pool        盲模型读出混合先验：s=λ*·ranknorm(log1p(deg_blind[可见端]))+(1-λ*)·ranknorm(learned)
              λ* 在验证天跨界边上从{0.3,0.5,0.7,0.9,1.0}选择（预注册网格）
  degfall     无学习先验 score=log1p(deg_blind[可见端])（强制必比基线；SimpleCount≡此臂，见论文声明）
评估集：跨界边（恰一端∈S）测试真实边 vs 跨界未连对（排除全历史），与 smoke3 协议逐位一致。
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, '/root/autodl-tmp/Euler-main')
sys.path.insert(0, '/root/autodl-tmp/Euler-main/benchmarks')
sys.path.insert(0, '/root/autodl-tmp/scripts')
from models.euler_serial import EulerGCN  # noqa: E402
from smoke1_masking import load_enron10, community_mask, random_mask, \
    neg_sample_visible  # noqa: E402


# ---------------- 手写 GRU（可插 mask 门控），与 nn.GRU 单层同构 ----------------
class MaskedGRU(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim, mode='freeze'):
        super().__init__()
        self.mode = mode
        self.W_ir = nn.Parameter(torch.empty(hidden_dim, feat_dim))
        self.W_iz = nn.Parameter(torch.empty(hidden_dim, feat_dim))
        self.W_in = nn.Parameter(torch.empty(hidden_dim, feat_dim))
        self.W_hr = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hz = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W_hn = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.b_ir = nn.Parameter(torch.zeros(hidden_dim))
        self.b_iz = nn.Parameter(torch.zeros(hidden_dim))
        self.b_in = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hr = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hz = nn.Parameter(torch.zeros(hidden_dim))
        self.b_hn = nn.Parameter(torch.zeros(hidden_dim))
        self.delta = nn.Parameter(torch.tensor(0.0))  # decay 率（gate_dec 用）
        self.lin = nn.Linear(hidden_dim, out_dim)
        for w in (self.W_ir, self.W_iz, self.W_in, self.W_hr, self.W_hz, self.W_hn):
            nn.init.orthogonal_(w)
        self.hidden_dim, self.feat_dim = hidden_dim, feat_dim

    def forward(self, xs, m_vis):
        """xs (T,N,F), m_vis (N,) ∈{0,1}：可见=1。返回 (T,N,out_dim)"""
        T, N, _ = xs.shape
        h = xs.new_zeros(N, self.hidden_dim)
        m_vis = m_vis.unsqueeze(1)  # (N,1) 便于与 (N,h) 广播
        outs = []
        if self.mode == 'dec':
            decay = torch.exp(-nn.functional.softplus(self.delta))
        for t in range(T):
            x = xs[t]
            r = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h @ self.W_hr.t() + self.b_hr)
            z = torch.sigmoid(x @ self.W_iz.t() + self.b_iz + h @ self.W_hz.t() + self.b_hz)
            n = torch.tanh(x @ self.W_in.t() + self.b_in + r * (h @ self.W_hn.t() + self.b_hn))
            h_new = (1 - z) * n + z * h
            if self.mode == 'frz':
                h = m_vis * h_new + (1 - m_vis) * h
            else:  # dec：盲节点记忆指数衰减
                h = m_vis * h_new + (1 - m_vis) * decay * h
            outs.append(h)
        return self.lin(torch.stack(outs))


class GateEuler(nn.Module):
    """EulerGCN 的 gate 位置变体：GCN 编码同款，GRU 换 MaskedGRU；decode 同款内积"""
    def __init__(self, x_dim, h_dim, z_dim, mode='freeze'):
        super().__init__()
        from models.euler_serial import GAE
        self.gcn = GAE(x_dim, hidden_dim=h_dim, embed_dim=h_dim)
        self.gru = MaskedGRU(h_dim, h_dim, z_dim, mode=mode)
        self.neg_weight = 0.5
        self.drop = nn.Dropout(0.05)
        self.sig = nn.Sigmoid()

    def forward(self, x, eis, mask_fn, m_vis):
        embeds = []
        for i in range(len(eis)):
            z = self.gcn(x, mask_fn(i), None)
            embeds.append(z)
        return self.gru(torch.tanh(torch.stack(embeds)), m_vis)

    def decode(self, src, dst, z):
        dot = (self.drop(z[src]) * self.drop(z[dst])).sum(dim=1)
        return self.sig(dot)

    def loss_fn(self, ts, fs, zs):
        EPS = 1e-6
        tot = 0.0
        for p, n_, z in zip(ts, fs, zs):
            pos_loss = -torch.log(self.decode(p[0], p[1], z) + EPS).mean()
            neg_loss = -torch.log(1 - self.decode(n_[0], n_[1], z) + EPS).mean()
            tot = tot + (1 - self.neg_weight) * pos_loss + self.neg_weight * neg_loss
        return tot / len(ts)


def ranknorm(v):
    """日内 rank 归一到 [0,1]（平手取平均秩）"""
    v = np.asarray(v, dtype=float)
    order = v.argsort().argsort().astype(float)
    return order / max(len(v) - 1, 1)


def ranknorm_joint(st, sf):
    """pos+neg 合并样本上联合 rank 归一（保组间序关系，AUC 语义正确）"""
    allv = np.concatenate([np.asarray(st, float), np.asarray(sf, float)])
    r = ranknorm(allv)
    return r[:len(st)], r[len(st):]


def train_model(arm, eis_in, tr_edges_in, N, x, end_tr, args, seed, S=None):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    if arm in ('full', 'blind', 'input'):
        model = EulerGCN(x.size(1), h_dim=args.hidden, z_dim=args.zdim,
                         gru_hidden_units=1, use_w=True)
    elif arm in ('gate_frz', 'gate_dec'):
        model = GateEuler(N, args.hidden, args.zdim,
                          mode='frz' if arm == 'gate_frz' else 'dec')
    opt = Adam(model.parameters(), lr=args.lr)
    m_vis = None
    if arm in ('gate_frz', 'gate_dec'):
        mv = torch.ones(N)
        mv[torch.tensor(S)] = 0.0  # 盲位=0
        m_vis = mv
    best, best_state, no_imp = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        if arm in ('gate_frz', 'gate_dec'):
            zs = model(x, eis_in, lambda i: eis_in[i], m_vis)[:end_tr]
        else:
            zs = model(x, eis_in, lambda i: eis_in[i])[:end_tr]
        p, n = [], []
        for t in range(end_tr):
            pe = tr_edges_in[t]
            p.append(pe)
            n.append(neg_sample_visible(pe, N, max(pe.size(1), 1), np.arange(N), rng))
        loss = model.loss_fn(p, n, zs)
        loss.backward()
        opt.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        with torch.no_grad():
            model.eval()
            if arm in ('gate_frz', 'gate_dec'):
                zs = model(x, eis_in, lambda i: eis_in[i], m_vis)[:end_tr]
            else:
                zs = model(x, eis_in, lambda i: eis_in[i])[:end_tr]
            st = torch.cat([model.decode(pp[0], pp[1], zz) for pp, zz in zip(p, zs)])
            sf = torch.cat([model.decode(nn_[0], nn_[1], zz) for nn_, zz in zip(n, zs)])
            lab = np.concatenate([np.ones(st.size(0)), np.zeros(sf.size(0))])
            sc = torch.cat([st, sf]).numpy()
            avg = average_precision_score(lab, sc) + roc_auc_score(lab, sc)
        if avg > best:
            best, best_state, no_imp = avg, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= args.patience and ep > 100:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def paired_bootstrap_diff(pos_a, neg_a, pos_b, neg_b, n_boot=2000, seed=0):
    """配对bootstrap：AUC(a)−AUC(b) 的95%CI（对同一样本对重采样）"""
    rng = np.random.RandomState(seed)
    sa = np.concatenate([pos_a, neg_a]); sb = np.concatenate([pos_b, neg_b])
    lab = np.concatenate([np.ones(len(pos_a)), np.zeros(len(neg_a))])
    n = len(sa)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        l = lab[idx]
        pa, na_ = sa[idx][l == 1], sa[idx][l == 0]
        pb, nb_ = sb[idx][l == 1], sb[idx][l == 0]
        if len(np.unique(l)) < 2:
            continue
        diffs.append(roc_auc_score(l, sa[idx]) - roc_auc_score(l, sb[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def paired_bootstrap_auc(pos_s, neg_s, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    P, Nn = len(pos_s), len(neg_s)
    aucs = []
    all_s = np.concatenate([pos_s, neg_s])
    lab = np.concatenate([np.ones(P), np.zeros(Nn)])
    for _ in range(n_boot):
        idx = rng.randint(0, P + Nn, P + Nn)
        aucs.append(roc_auc_score(lab[idx], all_s[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.005)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--zdim', type=int, default=16)
    ap.add_argument('--test-ts', type=int, default=3)
    ap.add_argument('--seed-offset', type=int, default=0)
    ap.add_argument('--data', default='/root/autodl-tmp/Euler-main/benchmarks/data/enron10/adj_orig_dense_list.pickle')
    ap.add_argument('--mode', required=True, help='community_0.4|random_0.4|community_0.2')
    ap.add_argument('--onset', type=int, default=0,
                    help='>0 时为晚发盲区：第 onset 天起 S 关联边才被遮蔽（此前观测保留，gate 臂记忆有前史可护）')
    ap.add_argument('--sigma-grid', action='store_true',
                    help='pool 的 λ×σ 联合网格（σ=+1/-1 读出符号翻转），验证天选择；否则仅 λ 网格')
    ap.add_argument('--out', default='/root/autodl-tmp/results/r1/e1_module1')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026)  # 固定 load_enron10→edge_tvt_split 的 randperm：切分可复现，E1/E1b 同评估集

    rate = float(args.mode.split('_')[1])
    is_comm = args.mode.startswith('community')
    onset = args.onset
    eis_full, splits, N = load_enron10(args.data)
    T = len(eis_full)
    end_tr = T - args.test_ts
    hist = set()
    for t in range(T):
        for u, v in eis_full[t].t().tolist():
            hist.add((min(u, v), max(u, v)))

    if is_comm:
        S, meta = community_mask(eis_full, N, rate)
    else:
        S, meta = random_mask(N, rate, np.random.RandomState(99))
    Smask = torch.zeros(N, dtype=torch.bool)
    Smask[torch.tensor(S)] = True
    Slist = [i for i in range(N) if Smask[i]]
    Vlist = [i for i in range(N) if not Smask[i]]
    def keep_fn_t(t, e):
        if t < onset:
            return torch.ones(e.size(1), dtype=torch.bool)
        return ~(Smask[e[0]] | Smask[e[1]])
    eis_blind = [eis_full[t][:, keep_fn_t(t, eis_full[t])] for t in range(T)]
    deg_blind = torch.zeros(N)
    for t in range(T):
        for u, v in eis_blind[t].t().tolist():
            deg_blind[u] += 1
            deg_blind[v] += 1

    # 跨界边评估集（同 smoke3：协议逐位一致；显式种子使评估集跨 run 可复现）
    np.random.seed(1234 + int(rate * 10) + (0 if is_comm else 1))
    eval_pos, eval_neg = [], []
    for t in range(end_tr, T):
        e = eis_full[t][:, splits[t][2]]
        cross = e[:, Smask[e[0]] ^ Smask[e[1]]]
        eval_pos.append((cross[0], cross[1], t))
        excl = set(hist)
        negs = []
        while len(negs) < max(cross.size(1), 1):
            u = Vlist[np.random.randint(len(Vlist))]
            v = Slist[np.random.randint(len(Slist))]
            key = (min(u, v), max(u, v))
            if key not in excl:
                excl.add(key)
                negs.append((u, v))
        arr = np.array(negs)
        eval_neg.append((torch.tensor(arr[:, 0]).long(), torch.tensor(arr[:, 1]).long(), t))

    # 验证天跨界边（λ* 选择用，预注册协议）
    va_pos, va_neg = [], []
    for t in range(end_tr - 2, end_tr):
        e = eis_full[t][:, splits[t][1]]
        cross = e[:, Smask[e[0]] ^ Smask[e[1]]]
        if cross.size(1) == 0:
            continue
        va_pos.append((cross[0], cross[1], t))
        excl = set(hist)
        negs = []
        while len(negs) < cross.size(1):
            u = Vlist[np.random.randint(len(Vlist))]
            v = Slist[np.random.randint(len(Slist))]
            key = (min(u, v), max(u, v))
            if key not in excl:
                excl.add(key)
                negs.append((u, v))
        arr = np.array(negs)
        va_neg.append((torch.tensor(arr[:, 0]).long(), torch.tensor(arr[:, 1]).long(), t))

    tr_full = [eis_full[t][:, splits[t][0]] for t in range(end_tr)]
    tr_blind = [tr_full[t][:, keep_fn_t(t, tr_full[t])] for t in range(end_tr)]

    outp = os.path.join(args.out, f'results_{args.mode}{"_onset%d" % onset if onset else ""}.jsonl')
    resf = open(outp, 'a')
    tag_extra = f'_onset{onset}' if onset else ''
    for seed in range(args.seed_offset, args.seed_offset + 3):
        t0 = time.time()
        rec = {'mode': args.mode, 'rate': rate, 'onset': onset, 'seed': seed,
               'n_blind': len(S),
               'n_pos': int(sum(p[0].size(0) for p in eval_pos)),
               'n_pos_va': int(sum(p[0].size(0) for p in va_pos))}
        x_eye = torch.eye(N)
        obs = torch.zeros(N, 2)
        obs[:, 0] = 1.0
        obs[torch.tensor(S), 0] = 0.0
        obs[torch.tensor(S), 1] = 1.0
        x_mask = torch.cat([x_eye, obs], dim=1)
        mv = torch.ones(N)
        mv[torch.tensor(S)] = 0.0  # 盲位=0
        arms_tr = {}
        arms_tr['full'] = train_model('full', eis_full, tr_full, N, x_eye, end_tr, args, seed)
        arms_tr['blind'] = train_model('blind', eis_blind, tr_blind, N, x_eye, end_tr, args, seed)
        # input 臂：mask 特征通道训练（同 smoke3 maskcond 协议），同评估集复跑对照
        arms_tr['input'] = train_model('input', eis_blind, tr_blind, N, x_mask, end_tr, args, seed)
        arms_tr['gate_frz'] = train_model('gate_frz', eis_blind, tr_blind, N, x_eye, end_tr, args, seed, S=S)
        arms_tr['gate_dec'] = train_model('gate_dec', eis_blind, tr_blind, N, x_eye, end_tr, args, seed, S=S)

        # 前向缓存各臂分数（学习部分）
        scores = {}  # tag -> (st, sf) numpy
        with torch.no_grad():
            for tag, m in arms_tr.items():
                zs = (m(x_mask, eis_blind, lambda i: eis_blind[i])
                      if tag == 'input' else
                      (m(x_eye, eis_blind, lambda i: eis_blind[i], mv)
                       if tag in ('gate_frz', 'gate_dec') else
                       m(x_eye, eis_full if tag == 'full' else eis_blind,
                         lambda i: (eis_full if tag == 'full' else eis_blind)[i])))
                st_all, sf_all = [], []
                for pos, neg in zip(eval_pos, eval_neg):
                    if pos[0].size(0) == 0:
                        continue
                    z = zs[pos[2]]
                    st_all.append(m.decode(pos[0], pos[1], z).numpy())
                    sf_all.append(m.decode(neg[0], neg[1], z).numpy())
                scores[tag] = (np.concatenate(st_all), np.concatenate(sf_all))

        # degfall 无学习分数（与评估集同序构造）
        st_d, sf_d = [], []
        for pos, neg in zip(eval_pos, eval_neg):
            if pos[0].size(0) == 0:
                continue
            for e_, acc in ((pos, st_d), (neg, sf_d)):
                u, v = e_[0].numpy(), e_[1].numpy()
                vis = np.where(Smask[u].numpy(), v, u)
                acc.append(np.log1p(deg_blind[vis].numpy()))
        scores['degfall'] = (np.concatenate(st_d), np.concatenate(sf_d))

        # 各臂 AUC + CI
        lab = np.concatenate([np.ones(len(scores['degfall'][0])),
                              np.zeros(len(scores['degfall'][1]))])
        for tag in ('full', 'blind', 'input', 'gate_frz', 'gate_dec', 'degfall'):
            st, sf = scores[tag]
            lo, hi = paired_bootstrap_auc(st, sf)
            rec[tag] = {'auc': float(roc_auc_score(lab, np.concatenate([st, sf]))),
                        'ap': float(average_precision_score(lab, np.concatenate([st, sf]))),
                        'auc_ci': [round(lo, 3), round(hi, 3)]}

        # pool 第五臂：λ*（及可选 σ 符号）在验证天跨界边选择；无验证跨界边则固定 0.8
        lam_grid = (0.3, 0.5, 0.7, 0.9, 1.0)
        sig_grid = (1, -1) if args.sigma_grid else (1,)
        lam_star, sig_star, va_ap_table = 0.8, 1, {}
        if len(va_pos) and sum(p[0].size(0) for p in va_pos) > 0:
            with torch.no_grad():
                m_blind = arms_tr['blind']
                zs_b = m_blind(x_eye, eis_blind, lambda i: eis_blind[i])
                vst, vsf = [], []
                for pos, neg in zip(va_pos, va_neg):
                    z = zs_b[pos[2]]
                    vst.append(m_blind.decode(pos[0], pos[1], z).numpy())
                    vsf.append(m_blind.decode(neg[0], neg[1], z).numpy())
                vst = np.concatenate(vst); vsf = np.concatenate(vsf)
            # 验证天 degfall 分数
            vd_pos, vd_neg = [], []
            for pos, neg in zip(va_pos, va_neg):
                for e_, acc in ((pos, vd_pos), (neg, vd_neg)):
                    u, v = e_[0].numpy(), e_[1].numpy()
                    vis = np.where(Smask[u].numpy(), v, u)
                    acc.append(np.log1p(deg_blind[vis].numpy()))
            vd_pos = np.concatenate(vd_pos); vd_neg = np.concatenate(vd_neg)
            vlab = np.concatenate([np.ones(len(vst)), np.zeros(len(vsf))])
            vpr_p, vpr_f = ranknorm_joint(vd_pos, vd_neg)
            vlr_p, vlr_f = ranknorm_joint(vst, vsf)
            for sig in sig_grid:
                vlr_eff_p = vlr_p if sig == 1 else (1.0 - vlr_p)
                vlr_eff_f = vlr_f if sig == 1 else (1.0 - vlr_f)
                for lam in lam_grid:
                    sp = lam * vpr_p + (1 - lam) * vlr_eff_p
                    sn = lam * vpr_f + (1 - lam) * vlr_eff_f
                    va_ap_table[f'sig{sig}_lam{lam}'] = float(
                        average_precision_score(vlab, np.concatenate([sp, sn])))
            best_key = max(va_ap_table, key=va_ap_table.get)
            sig_star = int(best_key.split('_')[0][3:])
            lam_star = float(best_key.split('_lam')[1])
        rec['lam_star'] = lam_star
        rec['sig_star'] = sig_star
        rec['va_ap_table'] = {str(k): round(v, 4) for k, v in va_ap_table.items()}
        st_l, sf_l = scores['blind']
        st_p, sf_p = scores['degfall']
        pr_p, pr_f = ranknorm_joint(st_p, sf_p)
        lr_p, lr_f = ranknorm_joint(st_l, sf_l)
        lr_p = lr_p if sig_star == 1 else (1.0 - lr_p)
        lr_f = lr_f if sig_star == 1 else (1.0 - lr_f)
        sp_pool = lam_star * pr_p + (1 - lam_star) * lr_p
        sf_pool = lam_star * pr_f + (1 - lam_star) * lr_f
        lo, hi = paired_bootstrap_auc(sp_pool, sf_pool)
        rec['pool'] = {'auc': float(roc_auc_score(lab, np.concatenate([sp_pool, sf_pool]))),
                       'ap': float(average_precision_score(lab, np.concatenate([sp_pool, sf_pool]))),
                       'auc_ci': [round(lo, 3), round(hi, 3)]}

        # 配对差分：pool vs degfall（Q3/Q5 学习增量）、pool vs blind（R1 双审 C1 补测：
        # blind 自然零假设反事实，未检验前 GO 不可归因模块①）、gate* vs blind、pool vs gate*
        stg = scores['gate_frz']; sgd = scores['gate_dec']
        rec['diff_pool_vs_degfall'] = [round(x, 4) for x in
                                       paired_bootstrap_diff(sp_pool, sf_pool, st_p, sf_p)]
        rec['diff_pool_vs_blind'] = [round(x, 4) for x in
                                     paired_bootstrap_diff(sp_pool, sf_pool, st_l, sf_l)]
        rec['diff_gatefrz_vs_blind'] = [round(x, 4) for x in
                                        paired_bootstrap_diff(stg[0], stg[1], scores['blind'][0], scores['blind'][1])]
        rec['diff_pool_vs_gatefrz'] = [round(x, 4) for x in
                                       paired_bootstrap_diff(sp_pool, sf_pool, stg[0], stg[1])]
        rec['sec'] = round(time.time() - t0, 1)
        resf.write(json.dumps(rec) + '\n')
        resf.flush()
        print(f"[{args.mode} seed{seed}] " +
              " ".join(f"{t}:AUC={rec[t]['auc']:.4f}" for t in
                       ('full', 'blind', 'input', 'gate_frz', 'gate_dec', 'degfall', 'pool')) +
              f" lam*={lam_star} ({rec['sec']}s)", flush=True)
    resf.close()
    print('E1_DONE', args.mode, flush=True)


if __name__ == '__main__':
    main()
