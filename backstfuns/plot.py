import numpy as np
import pickle

def plot_backstepping_results(rep):
    with open('./backstres/results.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
        results = pickle.load(f)

    errs_class = results[0]
    errs_safe = results[1]
    errs_fb = results[2]
    sols_class = results[3]
    sols_safe = results[4]
    sols_fb = results[5]
    t = results[-1]
    t = np.asarray(t)

    # replace/remove results that include numerical instabilities
    errs_class = np.nan_to_num(errs_class,nan=1e-322)
    errs_safe = np.nan_to_num(errs_safe,nan=1e-322)
    errs_fb = np.nan_to_num(errs_fb,nan=1e-322)

    errs_class_new = [errc for errc in errs_class if (np.asarray(errc) < 1e10).all() and \
        not np.isnan(errc).any() and not np.isinf(errc).any()] #  = errs_class
    errs_safe_new = [errs for errs in errs_safe if (np.asarray(errs) < 1e10).all() and \
        not np.isnan(errs).any() and not np.isinf(errs).any()] #= errs_safe
    errs_fb_new = [errs for errs in errs_fb if (np.asarray(errs) < 1e10).all() and \
        not np.isnan(errs).any() and not np.isinf(errs).any()] # = errs_fb

    err_norms_class = [np.sqrt(np.sum(error[:,0:3]**2,1)) for error in errs_class_new]
    err_norms_safe = [np.sqrt(np.sum(error[:,0:3]**2,1)) for error in errs_safe_new]
    err_norms_fb = [np.sqrt(np.sum(error[:,0:3]**2,1)) for error in errs_fb_new]

    median_err_class = np.median(err_norms_class,0)
    median_err_safe = np.median(err_norms_safe,0)
    median_err_fb = np.median(err_norms_fb,0)

    print('Median L2 tracking error with robust GP bound (our approach): %d' % median_err_safe.mean())

    print('Median L2 tracking error with vanilla GP: %d' % median_err_class.mean())

    print('Median L2 tracking error with fully Bayesian GP: %d' % median_err_fb.mean())

    import matplotlib.pyplot as plt
    from matplotlib import rc, rcParams

    rc('font', size=30)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    fig = plt.figure(num=None, figsize=(12, 5), dpi=80, facecolor='w', edgecolor='k')

    ax = fig.add_subplot()
    ax.grid(False)
    ax.set_axisbelow(True)
    plt.gcf().subplots_adjust(bottom=0.15)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.15)

    ax.plot(t, median_err_safe, 'b', label='Our approach', linewidth=6.0)
    ax.plot(t, median_err_class, 'r', label='GP', linewidth=6.0)
    ax.plot(t, median_err_fb, 'g', label='Full Bayes', linewidth=6.0)

    if rep > 1:
        lwdecile_class = np.quantile(err_norms_class,0.1,axis=0)
        updecile_class = np.quantile(err_norms_class,0.9,axis=0)

        lwdecile_safe = np.quantile(err_norms_safe,0.1,axis=0)
        updecile_safe = np.quantile(err_norms_safe,0.9,axis=0)

        lwdecile_fb = np.quantile(err_norms_fb,0.1,axis=0)
        updecile_fb = np.quantile(err_norms_fb,0.9,axis=0)


        ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([updecile_class,
                                lwdecile_class[::-1]]),
                 alpha=.2, fc='r', ec='None') #, label=r'$\pm \tilde{\beta} \tilde{\sigma}(x)$')
        ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([updecile_safe,
                                lwdecile_safe[::-1]]),
                 alpha=.2, fc='b', ec='None') #, label=r'$\pm$')
        ax.fill(np.concatenate([t, t[::-1]]),
                 np.concatenate([updecile_fb,
                                lwdecile_fb[::-1]]),
                 alpha=.2, fc='g', ec='None') #, label=r'$\pm$')

    ax.set_xlabel('Time $t$ (sec)')
    ax.set_ylabel(r'Norm of error $\vert \boldsymbol{e} \vert$')
    ylim_UB = 37 #4 ** (np.ceil(np.log2(np.max(updecile_class))))
    ylim_LB = 1e-30 #max(-2 ** (np.ceil(np.log2(np.abs(np.min(lwdecile_safe))))),1e-500)
    ax.set_ylim(ylim_LB, ylim_UB)
    ax.set_xlim(t.min(), t.max())
    ax.tick_params(
        axis='both',  # changes apply to both axes
        which='both',
        top=False)  # labels along the bottom edge are off
    ax.legend(loc='upper right', ncol=3, prop={'size': 26})
    # handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()

    # plt.savefig('../../../images/control_errs.pdf', format='pdf')

    plt.show()

