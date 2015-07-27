import os
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
width = 3.4
height = 3.0
dpi = 300
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "bold",
    "figure.dpi": dpi,
    "figure.figsize": (width, height),
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
    })
import math
import numpy
from pycbc.plot import plot_utils

def auto_pointsize(N):
    """
    Auto scales a point size based on the given number of points.
    """
    return 9. * numpy.log10(5e5)/numpy.log10(N) - 8

def plot_effectualness(results, xarg, xlabel, yarg, ylabel,
        tmplt_label='', inj_label='', logx=False, logy=False,
        plot_templates=False, templates=[],
        xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
        ptsize=None, cmap=pyplot.cm.Reds, bkgclr='k', dpi=300):
    
    fig = pyplot.figure(dpi=dpi)
    ax = fig.add_subplot(111, axisbg=bkgclr)
    plot_data = {}
    plot_data['args'] = (xarg, yarg)
    xvals = numpy.array([plot_utils.get_arg(res, xarg) for res in results])
    yvals = numpy.array([plot_utils.get_arg(res, yarg) for res in results])
    zvals = numpy.array([res.effectualness for res in results])

    # if nothing to plot, just create an empty plot and return
    if xvals.size == 0:
        plot_utils.empty_plot(ax, "No points to plot \mathcal{E}")
        return fig, None, None

    sort_idx = zvals.argsort()[::-1]
    xvals = xvals[sort_idx]
    yvals = yvals[sort_idx]
    zvals = zvals[sort_idx]
    plot_data['data'] = (xvals, yvals, zvals)

    if ptsize is None:
        ptsize = max(0.1, min(auto_pointsize(10), auto_pointsize(len(xvals))))
        
    sc = ax.scatter(xvals, yvals, edgecolors='none', c=zvals, s=ptsize,
        vmin=zmin, vmax=zmax, zorder=1, cmap=cmap)
    # make color bar above the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size = "5%", pad = 0.05)
    # ensure that there are only ever 6 ticks
    nticks = 5
    precision = 3 # the number of digits to print after the decimal
    if zmin is None:
        zmin = min(zvals)
    if zmax is None:
        zmax = max(zvals)
    dz = (zmax - zmin)/float(nticks)
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal',
        ticks=[round(zmin + (ii*dz), precision) for ii in range(nticks+1)])
    if tmplt_label or inj_label:
        lbl = ' '.join([tmplt_label.strip(), inj_label.strip()])
    else:
        lbl = ''
    cb.ax.set_xlabel('$\mathcal{E}_{\mathrm{%s}}$' % lbl, labelpad=-35)
    cb.ax.xaxis.set_ticks_position('top')

    # set the axis limits
    if xmin is None:
        if logx:
            xmin = xvals.min()*10**(-0.1)
        else:
            x_range = xvals.max() - xvals.min()
            xmin = xvals.min() - x_range/20.
    if xmax is None:
        if logx:
            xmax = xvals.max()*10**(0.1)
        else:
            x_range = xvals.max() - xvals.min()
            xmax = xvals.max() + x_range/20.
    if ymin is None:
        if logy:
            ymin = yvals.min()*10**(-0.1)
        else:
            y_range = yvals.max() - yvals.min()
            ymin = yvals.min() - y_range/20.
    if ymax is None:
        if logy:
            ymax = yvals.max()*10**(0.1)
        else:
            y_range = yvals.max() - yvals.min()
            ymax = yvals.max() + y_range/20.

    if plot_templates:
        xvals = [plot_utils.get_arg(tmplt, xarg) for tmplt in templates]
        yvals = [plot_utils.get_arg(tmplt, yarg) for tmplt in templates]
        ax.scatter(xvals, yvals, edgecolors='g', c='g', marker='x', s=40,
            zorder=10)
        plot_data['templates'] = (xvals, yvals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx and logy:
        ax.loglog()
    elif logx:
        ax.semilogx()
    elif logy:
        ax.semilogy()
    # set the axis limits
    if xmin != xmax:
        ax.set_xlim(xmin, xmax)
    if ymin != ymax:
        ax.set_ylim(ymin, ymax)

    # make axes white if the background is black
    if bkgclr == 'k' or bkgclr == 'black':
        ax.tick_params(axis='both', which='both', color='w')
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color('w')

    return fig, sc, plot_data



def plot_effectualness_cumhist(results, labels=[], colors=[],
        tmplt_label='', inj_label='',
        target_mismatch=0.97, xmin=None, xmax=None, ymin=None,
        ymax=None, logy=False, dpi=300):

    plot_data = {}
    fig = pyplot.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    for ii,result_group in enumerate(results):
        xvals = numpy.array([x.effectualness for x in result_group])
        if xvals.shape[0] == 0:
            # nothing to plot, continue
            continue
        xvals.sort()
        # yvals are the % of points with an effectualness <= each xval
        yvals = 100.*numpy.array([numpy.searchsorted(xvals, x, side='right') \
            for x in xvals]) / float(len(xvals))
        if labels != []:
            lbl = labels[ii]
        else:
            lbl = 'group %i' %(ii)
        if colors != []:
            clr = colors[ii]
        else:
            clr = None
            #if len(result_group) == 1:
            #    clrval = 0. 
            #else:
            #    clrval = ii/float(len(result_group)-1)
            #clr = pyplot.cm.brg(clrval)
        if clr is not None:
            ax.plot(xvals, yvals, color=clr, linewidth=1, zorder=ii, label=lbl)
        else:
            ax.plot(xvals, yvals, linewidth=1, zorder=ii, label=lbl)
        plot_data[lbl] = {}
        plot_data[lbl]['xvals'] = xvals
        plot_data[lbl]['yvals'] = yvals

    # if nothing was plotted, just create an empty plot and exit
    if plot_data == {}:
        plot_utils.empty_plot(ax) 
        return fig, plot_data
    if labels != []:
        ax.legend(loc='lower left')
    if logy:
        ax.semilogy()
    if tmplt_label or inj_label:
        lbl = ' '.join([tmplt_label.strip(), inj_label.strip()])
    else:
        lbl = ''
    ax.set_xlabel('$\mathcal{E}_{\mathrm{%s}}$' % lbl)
    ax.set_ylabel('$\% \leq \mathcal{E}$')
    # get axis limits
    plt_xmin, plt_xmax = ax.get_xlim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    plt_ymin, plt_ymax = ax.get_ylim()
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    # plot the target mismatch
    ax.plot([target_mismatch, target_mismatch], [plt_ymin, plt_ymax],
        color='darkorange', linestyle='--',
        linewidth=1, zorder=ii+1)
    # set the axis limits
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    
    ax.grid(which = 'both', zorder = 0)

    return fig, plot_data


def plot_effectualness_barcumhist(results, tmplt_label='', inj_label='',
        target_mismatch=0.97, nbins=20, xmin=None, xmax=None, ymin=None,
        ymax=None, logy=False, dpi=300):

    plot_data = {}
    fig = pyplot.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    if xmin is not None and xmax is not None:
        range = (xmin, xmax)
    else:
        range = None
    cnt, bins, _ = ax.hist([x.effectualness for x in results], bins=nbins,
        range=range, normed=True, cumulative=True, log=logy, zorder=1)
    plot_data['bins'] = bins
    plot_data['count'] = cnt
    if tmplt_label or inj_label:
        lbl = ','.join([tmplt_label.strip(), inj_label.strip()])
    else:
        lbl = ''
    ax.set_xlabel('$\mathcal{E}_{\mathrm{%s}}$' % lbl)
    ax.set_ylabel('cumulative fraction')
    # get axis limits
    plt_xmin, plt_xmax = ax.get_xlim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    plt_ymin, plt_ymax = ax.get_ylim()
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    # plot the target mismatch
    ax.plot([target_mismatch, target_mismatch], [plt_ymin, plt_ymax], 'r--',
        linewidth=2, zorder=2)
    # set the axis limits
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    
    ax.grid(which = 'both', zorder = 0)

    return fig, plot_data


def plotffcumhist(results, xarg, xlabel, target_mismatch = 0.97, xmin = None, xmax = None, ymin = None, ymax = None, title = None, dpi = 200):
    import bisect

    if xmin is not None and xmax is not None:
        range = (xmin, xmax)
    else:
        range = None
    plotvals = sorted([(plot_utils.get_arg(res, xarg), res.effectualness) for res in results])
    yvals = []
    ffs = []
    for _, ff in plotvals:
        bisect.insort(ffs, ff)
        cumPercent = 100*bisect.bisect_left(ffs, target_mismatch) / float(len(ffs))
        yvals.append(cumPercent)
    xvals = [x for x, _ in plotvals]

    fig = pyplot.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    ax.semilogy(xvals, yvals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'\%% $< %0.2f$' % target_mismatch)

    # set the axis limits
    plt_xmin, plt_xmax = ax.get_xlim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    plt_ymin, plt_ymax = ax.get_ylim()
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)

    if title is not None:
        ax.set_title(title)
    ax.grid(which = 'both')

    return fig

def plotffbinnedhist(results, xarg, xlabel, target_mismatch = 0.97, nbins = 50, xmin = None, xmax = None, ymin = None, ymax = None, title = None, dpi = 200):
    import bisect

    if xmin is not None and xmax is not None:
        range = (xmin, xmax)
    else:
        range = None
    plotvals = sorted([(plot_utils.get_arg(res, xarg), res.effectualness) for res in results])
    xvals = [x for x, _ in plotvals]
    ffs = [ff for _, ff in plotvals]
    xbins = numpy.linspace(plotvals[0][0], plotvals[-1][0], num = nbins+1)
    xbins[-1] *= 1.01
    yvals = numpy.zeros(nbins)
    xerr = numpy.zeros(nbins)
    yerr = numpy.zeros(nbins)
    for ii, min_val in enumerate(xbins[:-1]):
        max_val = xbins[ii+1]
        min_idx = bisect.bisect_left(xvals, min_val)+1
        max_idx = bisect.bisect_left(xvals, max_val)+1
        these_ffs = sorted(ffs[min_idx:max_idx])
        num_less_than_target = float(bisect.bisect_left(these_ffs, target_mismatch))
        yvals[ii] = 100*num_less_than_target / len(these_ffs)
        yerr[ii] = numpy.sqrt(num_less_than_target / len(these_ffs))  
        xerr[ii] = (max_val - min_val)/2.

    xvals = xbins[:-1] + numpy.diff(xbins)/2.

    fig = pyplot.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    ax.errorbar(xvals, yvals, xerr = xerr, yerr = yerr, marker = 'o')
    ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'\%% $< %0.2f$' % target_mismatch)

    # set the axis limits
    plt_xmin, plt_xmax = ax.get_xlim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    plt_ymin, plt_ymax = ax.get_ylim()
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)

    if title is not None:
        ax.set_title(title)
    ax.grid(which = 'both')

    return fig
