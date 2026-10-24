#!/usr/bin/env python3
"""Plot saved 4D-Var diagnostics without rerunning the model (headless)."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/swe-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

COLORS = dict(background='#b35832', analysis='#007f86', truth='#303c4b')


def read_csv(path):
    return np.atleast_1d(np.genfromtxt(path,delimiter=',',names=True))


def save(fig, out, name):
    for ext in ('png','pdf'):
        fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plot_results(output, movie=False):
    out = Path(output)
    summary = json.loads((out/'summary.json').read_text())
    data = np.load(out/'experiment.npz')
    metrics, costs, checks = [read_csv(out/name) for name in
                              ('trajectory_metrics.csv','cost_history.csv','gradient_check.csv')]
    cfg = summary['config']
    end = cfg['cycles']; times = data['times']; cutoff=times[end]
    truth,bg,an = [data[name] for name in ('truth','background','analysis')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold',
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig, axes = plt.subplots(2,3,figsize=(15,8.4),layout='constrained')
    for ax,comp,title in zip(axes[0,:2],('h','hu'),('Depth error','X-momentum error')):
        for name in ('background','analysis'):
            ax.semilogy(times,metrics[f'{name}_rmse_{comp}'],color=COLORS[name],lw=2.2,label=name.capitalize())
        ax.axvspan(cutoff,times[-1],color='#e1e8f0',alpha=.7,label='Free forecast')
        ax.axvline(cutoff,color='#697586',lw=1,ls='--')
        ax.set(xlabel='Time',ylabel='Domain RMSE',title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[0,2]
    for col,label,color in [('cost','Total','#303c4b'),('observation_cost','Observations','#007f86'),
                            ('background_cost','Background penalty','#b35832')]:
        ax.semilogy(costs['iteration'],np.where(costs[col]>0,costs[col],np.nan),label=label,color=color,lw=2)
    ax.set(xlabel='Accepted iteration',ylabel='Cost',title='Objective convergence')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0]
    ax.semilogy(costs['iteration'],costs['gradient_inf'],lw=2,color='#5b4a85')
    ax.set(xlabel='Accepted iteration',ylabel=r'$\|\nabla_v J\|_\infty$',title='Gradient in scaled controls')
    ax.grid(alpha=.2)
    ax=axes[1,1]
    locations=np.arange(2)
    for i,name in enumerate(('background','analysis')):
        values=[summary[key][name]/cfg['obs_std'] for key in ('training_obs_rmse','held_out_obs_rmse')]
        bars=ax.bar(locations+(i-.5)*.32,values,width=.32,color=COLORS[name],label=name.capitalize())
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.axhline(1,color='#4d5664',ls='--',lw=1,label='Noise standard deviation')
    ax.set(xticks=locations,xticklabels=['Assimilated sensors','Unused sensors'],ylabel='Residual RMS / observation σ',
           title='Fit and independent validation')
    ax.legend(fontsize=8,loc='upper center');ax.set_ylim(0,ax.get_ylim()[1]*1.45)
    ax=axes[1,2]
    for direction in np.unique(checks['direction']):
        rows=checks[checks['direction']==direction]
        ax.loglog(rows['epsilon'],np.maximum(rows['relative_error'],1e-15),'-o',ms=3,label=f'Direction {int(direction)+1}')
    ax.set(xlabel='Perturbation ε',ylabel='Central FD relative error',title='Discrete-adjoint verification')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8)
    gain=1-summary['forecast_rmse']['analysis'][0]/summary['forecast_rmse']['background'][0]
    fig.suptitle(f'Shallow-water 4D-Var  |  {cfg["nx"]} × {cfg["ny"]} cells, '
                 f'{cfg["sensors_side"]**2} height sensors\n'
                 f'Free-forecast depth RMSE reduction: {gain:.1%}',fontsize=16,fontweight='bold')
    save(fig,out,'var_4d_metrics')

    # All errors use one symmetric colour scale; corrections are never labelled as skill.
    frame_ids=[0,end,len(times)-1]
    fig,axes=plt.subplots(3,5,figsize=(16,9),layout='constrained')
    lo=min(np.min(truth[frame_ids,0]),np.min(bg[frame_ids,0]),np.min(an[frame_ids,0]))
    hi=max(np.max(truth[frame_ids,0]),np.max(bg[frame_ids,0]),np.max(an[frame_ids,0]))
    errmax=max(np.max(np.abs(bg[frame_ids,0]-truth[frame_ids,0])),np.max(np.abs(an[frame_ids,0]-truth[frame_ids,0])))
    idx=data['obs_indices'];sx=data['x'][idx%cfg['nx']];sy=data['y'][idx//cfg['nx']]
    titles=['Truth depth','Background depth','Analysis depth','Background − truth','Analysis − truth']
    for row,k in enumerate(frame_ids):
        values=[truth[k,0],bg[k,0],an[k,0],bg[k,0]-truth[k,0],an[k,0]-truth[k,0]]
        for col,val in enumerate(values):
            ax=axes[row,col]
            im=ax.imshow(val,origin='lower',extent=(0,1,0,1),cmap='viridis' if col<3 else 'RdBu_r',
                         vmin=lo if col<3 else -errmax,vmax=hi if col<3 else errmax)
            if row==0:ax.set_title(titles[col],fontsize=11)
            if col==0:
                label=['Initial state','Window endpoint','Free forecast'][row]
                ax.set_ylabel(f'{label}\nt = {times[k]:.2f}\ny')
            if row==2:ax.set_xlabel('x')
            if col==2:ax.scatter(sx,sy,s=5,c='white',alpha=.65,linewidths=0)
            if col>=3:
                rmse=np.sqrt(np.mean(val**2))
                ax.text(.03,.97,f'RMSE {rmse:.4f}',transform=ax.transAxes,va='top',fontsize=8,
                        bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
            if col==2:field_image=im
            if col==4:error_image=im
    fig.colorbar(field_image,ax=axes[:,:3],shrink=.6,label='Water depth')
    fig.colorbar(error_image,ax=axes[:,3:],shrink=.6,label='Depth error (shared scale)')
    fig.suptitle('Recovered initial state, assimilation endpoint, and independent forecast',fontsize=15,fontweight='bold')
    save(fig,out,'var_4d_snapshots')

    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for c,name in enumerate(('h','hu','hv')):
        ax=axes[c]
        for series in ('background','analysis'):
            ax.semilogy(times,metrics[f'{series}_rmse_{name}'],lw=2,color=COLORS[series],label=series.capitalize())
        ax.axvspan(cutoff,times[-1],color='#e1e8f0',alpha=.7)
        ax.axvline(cutoff,color='#697586',ls='--',lw=1)
        ax.set(xlabel='Time',ylabel='RMSE',title=['Depth h','X momentum hu','Y momentum hv'][c])
        ax.legend();ax.grid(alpha=.2)
    fig.suptitle('All state components; shaded region uses no observations',fontweight='bold')
    save(fig,out,'all_components_rmse')

    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for d in np.unique(checks['direction']):
        rows=checks[checks['direction']==d]
        axes[0].loglog(rows['epsilon'],rows['taylor_residual'],'-o',ms=3,label=f'Direction {int(d)+1}')
    rows=checks[checks['direction']==0]
    eps=rows['epsilon'];ref=rows['taylor_residual'][0]*(eps/eps[0])**2
    axes[0].loglog(eps,ref,'k--',label='O(ε²) reference')
    axes[0].set(xlabel='ε',ylabel='First-order Taylor remainder',title='Taylor test');axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2,which='both')
    axes[1].plot(checks['adjoint'],checks['finite_difference'],'.',color=COLORS['analysis'])
    lim=[min(checks['adjoint']),max(checks['adjoint'])]
    axes[1].plot(lim,lim,'k--',lw=1)
    axes[1].set(xlabel='Adjoint derivative',ylabel='Central finite difference',title='Signed derivative agreement')
    save(fig,out,'fd_adjoint_check')

    if movie:
        fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        fields=[truth[:,0],bg[:,0]-truth[:,0],an[:,0]-truth[:,0]]
        images=[]
        for c,(ax,vals,title) in enumerate(zip(axes,fields,['Truth','Background error','Analysis error'])):
            im=ax.imshow(vals[0],origin='lower',extent=(0,1,0,1),cmap='viridis' if c==0 else 'RdBu_r',
                         vmin=lo if c==0 else -errmax,vmax=hi if c==0 else errmax)
            images.append(im);ax.set(title=title,xlabel='x',ylabel='y')
            fig.colorbar(im,ax=ax,shrink=.75)
        title=fig.suptitle('')
        def update(k):
            for im,vals in zip(images,fields): im.set_data(vals[k])
            phase='Free forecast (no observations)' if k>end else 'Assimilation window'
            title.set_text(f'{phase}  |  t = {times[k]:.2f}')
            return images+[title]
        animation=FuncAnimation(fig,update,frames=len(times),interval=150,blit=False)
        animation.save(out/'var_4d_result.gif',writer=PillowWriter(fps=6),dpi=100)
        plt.close(fig)
    print(f'Plots saved in {out.resolve()}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='var_output')
    parser.add_argument('--movie',action='store_true',help='Also save a GIF animation')
    args=parser.parse_args()
    plot_results(args.output,args.movie)
