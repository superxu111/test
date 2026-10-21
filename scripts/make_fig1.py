# Fig1: PPD protocol schematic (mechanism diagram, no data). v2 layout, overlap-checked.
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT="/root/autodl-tmp/paper/figs/"
C={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","grey":"#7F7F7F","navy":"#1e293b","lblue":"#d6e8f5"}
plt.rcParams.update({"font.size":8})
fig,ax=plt.subplots(figsize=(7.6,3.1)); ax.set_xlim(0,104); ax.set_ylim(0,46); ax.axis("off")
def box(x,y,w,h,label,fc="white",ec=C["blue"],fs=8,lw=1.2,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5",fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,color=C["grey"],style="-",lw=1.2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,color=color,linestyle=style,lw=lw))
ax.text(52,44.3,"PPD: placement-aware reporting and audit protocol",fontsize=9,ha="center",fontweight="bold")
# (a) attacker placement
ax.text(11,40.0,"(a) attacker-chosen placement",fontsize=7,style="italic",ha="center")
box(1,26,16,11,"attacker\naction set $\\mathcal{A}$\n(S$_\\alpha$, A0–A4)",fc="#fdf1dd",ec=C["orange"],bold=True,fs=7.5)
for i,(lab,yy,wd) in enumerate([("high-vis",34,4.0),("mid-vis",30,2.2),("low-vis",26,8.0)]):
    ax.add_patch(Rectangle((20.5,yy),wd,2.8,fc=C["lblue"] if i<2 else "#f6c6b4",ec=C["navy"],lw=.8))
    ax.text(20.5+wd+0.7,yy+1.4,lab,fontsize=6.3,va="center")
arrow(17.6,31.5,20.0,31.5,color=C["orange"])
# (b) detector
ax.text(46,40.0,"(b) fixed detector, frozen artifacts",fontsize=7,style="italic",ha="center")
box(34,27,15,9,"temporal-graph\ndetector",fc="white",ec=C["blue"],bold=True)
box(34,15,15,6.5,"degree prior\n(degfall)",fc="#e8f4ee",ec=C["green"],fs=7.5)
box(56,19.5,9,9,"gate\nmixer $g$",fc="white",ec=C["blue"],bold=True)
arrow(49.5,31.5,55.4,26.0,color=C["blue"]); ax.text(50.2,30.3,"learned",fontsize=6.3,color=C["blue"])
arrow(49.5,18.2,55.4,21.5,color=C["green"]); ax.text(50.0,17.0,"prior",fontsize=6.3,color=C["green"])
arrow(65.6,24.0,72.6,24.0); ax.text(67.2,21.9,"score",fontsize=6.3)
# (c) three views (captions inside boxes)
ax.text(84,40.0,"(c) three report views",fontsize=7,style="italic",ha="center")
views=[("pooled","composition-weighted",C["blue"],32.0),("standardized","fixed stratum weights",C["orange"],24.0),("$F$ worst stratum","min over occupied",C["red"],16.0)]
for lab,sub,col,yy in views:
    box(73,yy,19,6.5,f"{lab}\n{sub}",fc="white",ec=col,fs=7.0)
    arrow(66.0,23.6,73.5,yy+3.2,lw=1.0)
# (d) audit
ax.text(66,10.9,"(d) audit of the evaluator",fontsize=7,style="italic",ha="center")
box(58,3,36,7,"information-symmetric audit A4$'$\n(auditor $\\leq$ attacker information)",fc="#eef3fb",ec=C["navy"],bold=True,fs=7.5)
arrow(86,15.6,86,10.4,color=C["navy"])
arrow(57.6,6.5,49.9,14.4,color=C["navy"],style="--")
ax.text(76.5,0.6,"mismatched audit inflates pooled advantage by +0.10–0.14",fontsize=6.3,color=C["red"],ha="center")
# evidence chips (bottom-left)
chips=[(1,17.0,"pooled spans 0.4933$\\to$0.0226 ($\\approx$22$\\times$)\nunder attacker-chosen placement"),
       (1,7.5,"pooled 0.8253  vs  $F$ = 0.0002\nat one best-response target")]
for x,y,txt in chips:
    ax.add_patch(FancyBboxPatch((x,y),29,7.0,boxstyle="round,pad=0.4",fc="#1e293b",ec="none"))
    ax.text(x+14.5,y+3.5,txt,ha="center",va="center",fontsize=5.6,color="#a7f3d0",family="monospace")
arrow(28.6,22.0,33.6,17.5,color=C["red"],lw=1.0)
fig.savefig(OUT+"fig1_protocol.pdf",bbox_inches="tight"); fig.savefig(OUT+"fig1_protocol.png",dpi=600,bbox_inches="tight")
print("Fig1 v2 done")
