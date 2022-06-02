# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import rc

def plot1D(X_data,y_data,X_test,y_pred, beta, sig_pred, beta_safe, sig_safepr, y_predfb, sig_prfb, f_true):
    y_pred = y_pred.detach()
    y_data = y_data.detach()

    # Plot the function, the prediction and the 95% confidence interval
    rc('font', size=50)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    
    fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_axisbelow(True)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
    plt.plot(X_data, y_data, 'kx', markersize=18, label='Observations',mew=5.0)
    plt.plot(X_test, f_true, 'k-', label=r'$f(x)$',linewidth=4.0)
    plt.plot(X_test, y_pred, 'r-.', label=r'$\mu_{\vartheta_0}$(x)',linewidth=4.0)
    plt.plot(X_test, y_predfb, 'g-.', label=r'$\mu_{{FB}}$(x)',linewidth=4.0)
    plt.fill(torch.cat([X_test, torch.flip(X_test,[0])]),
             torch.cat([y_pred - beta_safe * sig_safepr,
                            torch.flip(y_pred + beta_safe * sig_safepr,[0])]).detach(),
             alpha=.2, fc='b', ec='None', label= 'Our approach') # r'$\pm \bar{\beta}^{\frac{1}{2}} {\sigma}_{\vartheta^\prime}(x)$')
    plt.fill(torch.cat([X_test, torch.flip(X_test,[0])]),
             torch.cat([y_pred - beta * sig_pred,
                            torch.flip(y_pred + beta * sig_pred,[0])]).detach(),
             alpha=.2, fc='r', ec='None', label= 'Vanilla GP') # r'$\pm \beta^{\frac{1}{2}} \sigma_{\vartheta_0}(x)$')
    plt.fill(torch.cat([X_test, torch.flip(X_test, [0])]),
             torch.cat([y_predfb - beta * sig_prfb,
                        torch.flip(y_predfb + beta * sig_prfb, [0])]).detach(),
             alpha=.2, fc='g', ec='None', label='Full Bayes') #r'$\pm \beta^{\frac{1}{2}} \sigma_{{FB}}(x)$')
    plt.xlabel('$x$')
    if X_data.shape[0] < 3:
        plt.ylabel('$f(x)$')
    ylim_UB = 9.5 # 2**(torch.ceil(torch.log2(torch.max(y_pred+ beta_safe * sig_safepr)))+1)
    ylim_LB = -3.5 #-2**(torch.ceil(torch.log2(torch.abs(torch.min(y_pred- beta_safe * sig_safepr)))))
    plt.ylim(ylim_LB, ylim_UB)
    plt.xlim(X_test.min(),X_test.max())
    plt.tick_params(
        axis='both',          # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,
        top=False,         # ticks along the top edge are off
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    plt.tight_layout()

    # if X_data.shape[0] > 8:
    plt.legend(loc='upper left', ncol=3, prop={'size': 26})
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles,labels,loc='upper left', ncol=3, prop={'size': 26},frameon=False)
    ## save legend separately
    # leg = legend.figure
    # leg.canvas.draw()
    # bbox = legend.get_window_extent().transformed(leg.dpi_scale_trans.inverted())
    # leg.savefig('legend.pdf', format='pdf', bbox_inches=bbox)

    # plt.savefig('safegp_N' + X_data.shape[0].__str__() + '.pdf', format='pdf')
    
    plt.show()