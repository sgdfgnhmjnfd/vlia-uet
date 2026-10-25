"""Plot the README results figure from results/ladder_v1 and results/locked_test_final_v1 (run from repo root)."""
import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager as fm
try:
    for f in ['lmroman10-regular.otf','lmroman10-bold.otf']: fm.fontManager.addfont('/usr/share/texmf/fonts/opentype/public/lm/'+f)
    plt.rcParams['font.family']='Latin Modern Roman'
except OSError:
    plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'
def load(p, names):
    d={}
    for r in csv.DictReader(open(p)):
        if r['pipeline'] in names:
            d[(names[r['pipeline']],int(r['k']))]=(100*float(r['5way_acc']),20*float(r['5way_lift_ci_lo']),20*float(r['5way_lift_ci_hi']))
    return d
cv=load('results/ladder_v1/summary.csv',{'baseline':'old','nested_selected':'new'})
te=load('results/locked_test_final_v1/summary.csv',{'baseline':'old','frozen':'new'})
fig,axs=plt.subplots(1,2,figsize=(8,3.0),sharey=True)
ks=[2,4,6,8]; x=np.arange(4); w=0.36
for ax,d,t in [(axs[0],cv,'(a) Scene-grouped 8-fold CV (2,503 steps)'),(axs[1],te,'(b) Locked test, evaluated once (407 steps)')]:
    for i,(m,lab,c) in enumerate([('old','Baseline: SigLIP-B/16, raw','#9aa5b1'),('new','Ours: SigLIP-so400m + context subtraction','#d9480f')]):
        v=[d[(m,k)] for k in ks]; y=[a for a,_,_ in v]
        err=[[a-l for a,l,_ in v],[h-a for a,_,h in v]]
        ax.bar(x+(i-0.5)*w,y,w,yerr=err,capsize=2,color=c,label=lab,error_kw=dict(lw=0.8))
        for xi,yi in zip(x,y): ax.text(xi+(i-0.5)*w,5,f'{yi:.0f}',ha='center',va='center',fontsize=6.5,color='white',fontweight='bold')
    ax.axhline(20,ls='--',c='k',lw=0.9,label='Chance (20%)')
    ax.set_xticks(x); ax.set_xticklabels([f'k = {k}' for k in ks],fontsize=8)
    ax.set_title(t,fontsize=8.5); ax.set_ylim(0,47); ax.tick_params(labelsize=8)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_ylabel('5-way accuracy (%)',fontsize=8)
axs[1].legend(fontsize=8,loc='upper left',frameon=False)
plt.tight_layout(pad=0.3,w_pad=0.6); plt.savefig('docs/assets/results_5way.pdf'); plt.savefig('docs/assets/results_5way.png',dpi=200)
