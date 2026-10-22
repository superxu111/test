# Paper figures: placement-aware evaluation (WWW 2027 submission draft).
# Data (READ-ONLY): results/r3/e32_cas_stratified/, results/r3/e33_a4prime_adaptive/
# Outputs: paper/figs/fig2..fig4 (.pdf/.png). CPU only; no training; no writes under results/.
import json, collections, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"]="none"

R="/root/autodl-tmp/results/r3/"; OUT="/root/autodl-tmp/paper/figs/"
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False})
OI={"blue":"#0072B2","orange":"#E69F00","red":"#D55E00","green":"#009E73","grey":"#999999"}
def save(fig,name):
    fig.savefig(OUT+name+".pdf",bbox_inches="tight"); fig.savefig(OUT+name+".png",dpi=600,bbox_inches="tight"); fig.savefig(OUT+name+".svg",bbox_inches="tight"); plt.close(fig)

fam=["S_a0.25","S_a0.5","S_a0.75","S_a1.0","S_a1.5","S_am0.5","A0_uniform","A1_avoid_b1","A2_avoid_b2","A3_br_learn","A4_br_gate"]
rows=[json.loads(l) for l in open(R+"e32_cas_stratified/stratified_table.jsonl")]
agg=collections.defaultdict(list); lcells=collections.defaultdict(list)
for r in rows:
    if r.get("conv")!="asimpl": continue
    for sc in ("learn","gate"): agg[sc,r["strategy"]].append(r["metrics"][sc]["pooled"])
    lcells[r["strategy"]].append(r["metrics"]["learn"]["pooled"])
learn={s:st.mean(agg["learn",s]) for s in fam}; gate={s:st.mean(agg["gate",s]) for s in fam}
declared=fam[:-2]
mincell=min(min(v) for v in [lcells[s] for s in declared]); maxl=max(learn[s] for s in declared)
print("[Fig2] learn:",{s:round(learn[s],4) for s in fam})
print("[Fig2] gate :",{s:round(gate[s],4) for s in fam})
print(f"[Fig2] span max(learn)={maxl:.4f} mincell={mincell:.4f} ratio={maxl/mincell:.1f}")
fig,ax=plt.subplots(figsize=(6.4,2.6))
for s,v in lcells.items(): ax.scatter([s]*len(v),v,s=3,color=OI["grey"],alpha=.22,zorder=2)
ax.plot(fam,[learn[s] for s in fam],"o-",color=OI["red"],lw=1.4,ms=4,label="learned (no gating)",zorder=3)
ax.plot(fam,[gate[s] for s in fam],"s-",color=OI["blue"],lw=1.4,ms=4,label="degree-gated",zorder=3)
ax.set_ylabel("pooled AUC"); ax.set_ylim(-0.03,1.03)
ax.set_title("Attacker-chosen placement family, as-implemented visibility",fontsize=8)
a2=learn["A2_avoid_b2"]
ax.annotate(f"declared family span {maxl:.4f}$\\to${a2:.4f} ($\\approx${maxl/a2:.0f}$\\times$); lowest cell {mincell:.4f}",xy=(0.02,0.58),xycoords="axes fraction",fontsize=6.5)
ax.axvline(8.5,color="k",lw=0.9,ls="--",zorder=4)
ax.text(8.62,0.04,"single-round\nbest responses",fontsize=5.5)
ax.legend(frameon=False,fontsize=6.5,loc="upper right",bbox_to_anchor=(1.0,0.86)); ax.tick_params(axis="x",rotation=35)
save(fig,"fig2_placement_family")

e33=[json.loads(l) for l in open(R+"e33_a4prime_adaptive/results_e33.jsonl")]
J2=json.load(open(R+"e33_a4prime_adaptive/analysis_e33.json"))["J2_J4_adaptive_table"]
def seed_cells(target):
    d=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in e33:
        if r["conv"]=="asimpl" and r["target"]==target:
            for m in ("pooled","std","F"): d[m][r["seed"]].append(r["metrics"]["gate"][m])
    return d
mets=["pooled","std","F"]; targets=["gate_0.2","gate_0.5"]
for t in targets:
    d=seed_cells(t); cm={m:st.mean([x for v in d[m].values() for x in v]) for m in mets}
    print(f"[Fig3] {t} 15-cell:",{m:round(cm[m],4) for m in mets},"| J2_J4:",{("gate_"+m):round(J2[t]["gate_"+m],4) for m in mets},
          "| per-seed pooled:",{s:round(st.mean(d['pooled'][s]),4) for s in sorted(d['pooled'])})
fig,ax=plt.subplots(figsize=(4.8,2.5))
w=0.22; cols={"pooled":OI["blue"],"std":OI["orange"],"F":OI["red"]}
for i,t in enumerate(targets):
    d=seed_cells(t)
    for j,m in enumerate(mets):
        seeds=sorted(d[m]); pm=[st.mean(d[m][s]) for s in seeds]; x=i+(j-1)*w
        ax.bar(x,st.mean(pm),width=w*.9,color=cols[m],label=m if i==0 else None,zorder=3)
        ax.scatter([x]*len(pm),pm,s=8,color="k",zorder=4)
ax.set_xticks(range(len(targets))); ax.set_xticklabels(["target $g_{0.2}$","target $g_{0.5}$"])
ax.set_ylabel("gate AUC"); ax.set_ylim(0,1.0); ax.legend(frameon=False,fontsize=7,ncol=3)
ax.set_title("Three report views at two best-response targets",fontsize=8)
save(fig,"fig3_three_metrics")

J1=json.load(open(R+"e33_a4prime_adaptive/analysis_e33.json"))["J1_mismatch_quantified"]
fig,ax=plt.subplots(figsize=(3.6,2.5))
for i,s in enumerate(sorted(J1)):
    e=J1[s]["A4prime_gt0.5_vs_e25_A4ref"]; mu={"0":-0.1010,"1":-0.1367,"2":-0.1412}[s]; ci=e["ci95"]
    ax.bar(i,mu,width=.55,color=OI["green"],zorder=3)
    ax.errorbar(i,mu,yerr=[[mu-ci[0]],[ci[1]-mu]],fmt="none",ecolor="k",capsize=3,lw=1,zorder=4)
    ax.text(i,min(ci)-0.012,f"{mu:.4f}",ha="center",fontsize=7)
ax.axhline(0,color="k",lw=.8)
ax.set_xticks(range(len(J1))); ax.set_xticklabels([f"seed {s}" for s in sorted(J1)])
ax.set_ylabel(r"$\Delta$ pooled AUC (gate)"); ax.set_ylim(-0.19,0.03)
ax.set_title("Information-symmetric audit A4$'$ vs mismatched A4",fontsize=8)
save(fig,"fig4_a4prime_audit")
print("[Fig4] deltas:",{s:(J1[s]["A4prime_gt0.5_vs_e25_A4ref"]["mean"],J1[s]["A4prime_gt0.5_vs_e25_A4ref"]["ci95"]) for s in sorted(J1)})
print("DONE")
