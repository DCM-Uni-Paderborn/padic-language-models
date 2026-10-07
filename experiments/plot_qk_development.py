"""Scientific plot of every frozen readout's quality/access tradeoff."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.input;out=args.output
    if out.exists():raise FileExistsError('fresh figure output required')
    h=hashlib.sha256((p/'completed.json').read_bytes()).hexdigest()
    assert h=='2c39f405bfd0ded6e86850d3bd46320877a22e2ed98a2f02321cbb75196f4660'
    r=json.loads((p/'results.json').read_text());m=r['methods']
    colors={'initial_p2':'#777777','p2':'#0072B2','p3':'#E69F00','flat_euclidean':'#009E73','flat_dot':'#D55E00','gaussian':'#CC79A7'}
    labels={'initial_p2':'Initial binary','p2':'Trained binary','p3':'Trained ternary','flat_euclidean':'Flat Euclidean','flat_dot':'Flat restored dot','gaussian':'Gaussian selection'}
    markers={17:'o',29:'s',43:'^'}
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.7))
    for name,v in m.items():
        y=100*(v['perplexity_ratio']-1)
        if name.startswith('s') and name!='sink_recency':
            seed=int(name.split('_')[0][1:]);kind=name.split('_',1)[1]
            color,marker=colors[kind],markers[seed]
        else:
            color,marker={'full_control':('#333333','D'),'recency':('#111111','*'),
                'sink_recency':('#555555','P'),'angular':('#56B4E9','X')}[name]
        for ax,key in zip(axes,['affected_mean_logical_gqa_union','affected_projected_nrmse']):
            ax.scatter(v[key],y,c=color,marker=marker,s=62,edgecolors='white',linewidths=.65,zorder=3)
    for ax in axes:
        ax.set_yscale('symlog',linthresh=.05,linscale=1)
        ax.set_ylim(-.003,5.2);ax.set_yticks([0,.025,.05,.1,.25,.5,1,2,4])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.grid(alpha=.2);ax.set_ylabel('Perplexity change against stock (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_xlabel('Mean selected-position union per GQA group\n(logical count on affected queries)')
    axes[0].axvline(128,color='#777777',ls=':',lw=1);axes[0].set_xlim(120,330)
    axes[1].set_xlabel('Projected attention-output NRMSE\n(native BF16, affected queries)');axes[1].set_xlim(-.008,.27)
    fig.suptitle('Frozen first-layer development: quality, logical access and local error',fontsize=14,y=.96)
    fig.text(.5,.91,'512-token contexts • 128 keys/head • 64 windows / 58 articles • all declared seeds and controls',ha='center',fontsize=10)
    handles=[Line2D([],[],color=c,marker='o',ls='',label=labels[k]) for k,c in colors.items()]
    handles.extend(Line2D([],[],color=c,marker=mark,ls='',label=label) for label,c,mark in
        [('Dense / full control','#333333','D'),('Recency','#111111','*'),('Four sinks + recency','#555555','P'),('Angular','#56B4E9','X')])
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.5,.072),frameon=False,fontsize=9)
    fig.text(.5,.03,'Seed markers: 17 ○, 29 □, 43 △. Descriptive correlated data; no uncertainty interval. All real K/V remain retained.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.07,right=.985,bottom=.25,top=.86,wspace=.27)
    out.mkdir(parents=True);fig.savefig(out/'quality-access.png',dpi=180);fig.savefig(out/'quality-access.pdf');plt.close(fig)
    (out/'plot_qk_development.py').write_bytes(Path(__file__).read_bytes())
    (out/'manifest.json').write_text(json.dumps({'input_completion_sha256':h,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'plotted_methods':list(m),'full_control_is_dense_point':True,'scope':'Standalone scientific figure, all22 declared readouts, descriptive development'},indent=2)+'\n')


if __name__=='__main__':main()
