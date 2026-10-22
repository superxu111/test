# Fig1: PPD protocol schematic — drawn at true column width (3.5in) so all
# font sizes render at their nominal point size (>=7pt for chips).
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT="/root/autodl-tmp/paper/figs/"
C={"blue":"#0072B2","orange":"#E69F00","green":"#009E73","red":"#D55E00","grey":"#7F7F7F","navy":"#1e293b","lblue":"#d6e8f5"}
plt.rcParams.update({"font.size":6,"svg.fonttype":"none"})
fig,ax=plt.subplots(figsize=(3.5,1.95)); ax.set_xlim(0,100); ax.set_ylim(0,54); ax.axis("off")
def box(x,y,w,h,label,fc="white",ec=C["blue"],fs=5.2,lw=0.9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4",fc=fc,ec=ec,lw=lw))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs)
def arrow(x1,y1,x2,y2,color=C["grey"],style="-",lw=0.9):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=6,color=color,linestyle=style,lw=lw))
ax.text(50,51.5,"PPD: placement-aware reporting and audit",fontsize=6.2,ha="center",fontweight="bold")
# (a) placement
ax.text(11,46.6,"(a) attacker placement",fontsize=5,style="italic",ha="center")
box(1,31,17,12,"attacker\naction set\n(S$_\\alpha$, A0–A4)",fc="#fdf1dd",ec=C["orange"],fs=4.8)
for i,(lab,yy,wd) in enumerate([("high",38,3.4),("mid",34.5,1.9),("low",31,6.2)]):
    ax.add_patch(Rectangle((21.5,yy),wd,2.3,fc=C["lblue"] if i<2 else "#f6c6b4",ec=C["navy"],lw=.6))
    ax.text(21.5+wd+0.6,yy+1.15,lab,fontsize=4.6,va="center")
arrow(18.4,37,21.0,37,color=C["orange"])
# (b) detector
ax.text(43,46.6,"(b) frozen detector",fontsize=5,style="italic",ha="center")
box(35,31.5,12,10.5,"temporal-\ngraph\ndetector",fs=5.0)
box(35,23,12,6,"degree prior",fc="#e8f4ee",ec=C["green"],fs=5.2)
box(51.5,27,8,9,"gate\n$g$",fs=5.5)
arrow(47.4,37,51.0,33.6,color=C["blue"]); ax.text(47.6,38.8,"learned",fontsize=4.6,color=C["blue"])
arrow(47.4,26,51.0,29.4,color=C["green"]); ax.text(47.6,24.2,"prior",fontsize=4.6,color=C["green"])
arrow(59.9,31.5,64.2,31.5); ax.text(59.6,32.9,"score",fontsize=4.2)
# (c) three views
ax.text(82,46.6,"(c) three views",fontsize=5,style="italic",ha="center")
for lab,sub,col,yy in [("pooled","comp.-weighted",C["blue"],36.5),("standardized","fixed weights",C["orange"],28.0),("$F$ worst","min occupied",C["red"],19.5)]:
    box(65,yy,17,6.5,f"{lab}\n{sub}",fc="white",ec=col,fs=4.8)
    arrow(60.1,31,64.4,yy+3.2,lw=0.8)
# (d) audit
ax.text(64,13.3,"(d) audit",fontsize=5,style="italic",ha="center")
box(50,3,34,8.5,"A4$'$ audit\nauditor $\\leq$ attacker info",fc="#eef3fb",ec=C["navy"],fs=5.2)
arrow(73.5,19.0,73.5,12.0,color=C["navy"])
arrow(49.4,6.2,45.5,21.5,color=C["navy"],style="--")
ax.text(67,0.6,"audit mismatch hides $+0.316$ (reports $-0.006$)",fontsize=4.8,color=C["red"],ha="center")
# chips (single line, >=7pt)
chips=[(1,23.6,"0.4933→0.0251 (≈20×)"),(1,16.4,"0.8253 vs F=0.0002")]
for x,y,txt in chips:
    ax.add_patch(FancyBboxPatch((x,y),32,5.8,boxstyle="round,pad=0.3",fc="#1e293b",ec="none"))
    ax.text(x+17,y+2.9,txt,ha="center",va="center",fontsize=6.5,color="#a7f3d0",family="monospace")
arrow(33.6,26.5,38.6,25.6,color=C["red"],lw=0.9)
arrow(35.3,19.1,39.0,23.2,color=C["red"],lw=0.9)
fig.savefig(OUT+"fig1_protocol.pdf",bbox_inches="tight"); fig.savefig(OUT+"fig1_protocol.png",dpi=300,bbox_inches="tight"); fig.savefig(OUT+"fig1_protocol.svg",bbox_inches="tight")
print("Fig1 v5 done (true-size fonts)")
