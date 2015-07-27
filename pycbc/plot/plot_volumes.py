#! /usr/bin/env python

import sys
import lal

columnwidth = 3.4
width = 2*columnwidth
height = 3.5
from matplotlib import patches as mplpatches
from matplotlib import pyplot
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.weight": "bold",
    "figure.figsize": (width, height),
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    })
pyplot.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')

from pycbc.plot import plot_utils
from pycbc.plot import efficiency
import numpy
import cPickle
from optparse import OptionParser

#----------------------------------------------------------------------
#
#               Useful functions for making the plots
#
#----------------------------------------------------------------------

def get_log10_ticks(ticks):
    """
    Given some ticks from a log10 axis, converts them into 10^{**} form.

    Parameters
    ----------
    ticks: output of ax.get_(x|y)ticks()

    Returns
    -------
    new_ticks: list
        new tick values; can set the appropriate axis using ax.set_(x|y)ticks
    ticklabels: list
        new tick labels; can set the appropriate axis using
        ax.set_(x|y)ticklabels
    """
    min_tick = numpy.ceil(min(ticks))
    max_tick = numpy.floor(max(ticks))
    major_ticks = numpy.arange(min_tick, max_tick+1)
    new_ticks = numpy.array([coef*10**pwr for pwr in major_ticks[:-1] \
        for coef in range(1,10)] + [10**major_ticks[-1]])
    ticklabels = [tck % 1. == 0 and '$10^%i$' % tck or '' \
        for tck in numpy.log10(new_ticks)]
    return numpy.log10(new_ticks), ticklabels


def _plot_tiles(ax, zvals, plus_errs, minus_errs, phyper_cubes,
        xarg, xlabel, yarg, ylabel,
        colormap='hot', vmax=None, vmin=None,
        add_colorbar=False, cbformat=None,
        annotate=True, fontsize=8, print_relative_err=False, 
        logx=False, logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, add_clickables=True):
    """
    Base-level function to create a tiles plot.
    """
    if vmin is None:
        vmin = zvals.min()
    if vmax is None:
        vmax = zvals.max()
    if logz:
        zvals = numpy.log10(zvals)
        vmin = numpy.log10(vmin)
        vmax = numpy.log10(vmax)
    # check what the grayscale is on either end of the color maps; if vmin or
    # vmax is close to black, we'll bump them down/up slightly so as not to
    # blend with the background
    if False:
        target_min_gray = 0.05
        gray_precision = 0.01
        gray_frac_diff = 0.
        min_gray = plot_utils.get_color_grayscale(
            getattr(pyplot.cm, colormap)(0.))
        new_vmin = vmin
        # to prevent infinite loops, we'll give up after 100 steps
        step_count = 0
        while min_gray < target_min_gray and step_count < 100:
            # if the fractional difference is > 0, means we've gone too far to
            # white, so step back half
            if gray_frac_diff > 0:
                step_size = step_size/2.
            # otherwise, increase the normalization by 1.5; this has the effect
            # of sticking the minimum point a quarter of the way into the color
            # range
            else:
                step_size = -abs(vmax - new_vmin)/2. #abs(0.01*new_vmin)
            new_vmin = new_vmin + step_size 
            clrfac = (vmin - new_vmin)/(vmax - new_vmin)
            min_gray = plot_utils.get_color_grayscale(
                getattr(pyplot.cm, colormap)(clrfac))
            gray_frac_diff = (min_gray - target_min_gray)/target_min_gray
            step_count += 1
        vmin = new_vmin
        # ditto for vmax
        gray_frac_diff = 0.
        min_gray = plot_utils.get_color_grayscale(
            getattr(pyplot.cm, colormap)(1.))
        new_vmax = vmax
        step_count = 0
        while min_gray < target_min_gray and step_count < 100:
            # if the fractional difference is > 0, means we've gone too far to
            # white, so step back half
            if gray_frac_diff > 0:
                step_size = -step_size/2.
            # otherwise, increase the normalization by 1.5; this has the effect
            # of sticking the minimum point a quarter of the way into the color
            # range
            else:
                step_size = abs(new_vmax - vmin)/2. #abs(0.01*new_vmin)
            new_vmax = new_vmax + step_size 
            clrfac = (new_vmax - vmax)/(new_vmax - vmin)
            min_gray = plot_utils.get_color_grayscale(
                getattr(pyplot.cm, colormap)(clrfac))
            gray_frac_diff = (min_gray - target_min_gray)/target_min_gray
            step_count += 1
        vmax = new_vmax
    if annotate and (plus_errs is None or minus_errs is None):
        raise ValueError("annotate requires plus_errs and minus_errs")
    # if not annotating, we don't need the errors, so if they were passed as
    # None, just create some dummy arrays
    if plus_errs is None:
        plus_errs = numpy.zeros(len(phyper_cubes))
    if minus_errs is None:
        minus_errs = plus_errs

    # cycle over the cubes and zvals, creating each tile
    for Z, err_plus, err_minus, this_cube in zip(zvals, plus_errs, minus_errs,
            phyper_cubes):

        # get the x, y corners of the tile from the bounds
        xlow, xhigh = this_cube.get_bound(xarg)
        ylow, yhigh = this_cube.get_bound(yarg)
        x = numpy.array([xlow, xlow, xhigh, xhigh])
        if logx:
            x = numpy.log10(x)
        y = numpy.array([ylow, yhigh, yhigh, ylow])
        if logy:
            y = numpy.log10(y)

        # get the color to use for this cube
        clrfac = (Z - vmin)/(vmax - vmin)
        clr = getattr(pyplot.cm, colormap)(clrfac)

        # get the color to use for the text
        clr_grayscale = plot_utils.get_color_grayscale(clr)
        if annotate:
            if clr_grayscale < 0.35:
                txt_clr = 'w'
            else:
                txt_clr = 'k'
            # get the text to print; note that if we are using logz, we convert
            # Z back to normal here
            if logz:
                Z = 10**Z
            # because we are not printing units, we can just use
            # format_volume_text, for the text string, even though the zvals
            # may be gains and not volumes
            if Z != 0. and (Z >= 100000. or Z < 0.01):
                use_scin = True
            else:
                use_scin = False
            txt_str = efficiency.format_volume_text(Z, err_plus, err_minus,
                include_units=False, use_scientific_notation=use_scin,
                use_relative_err=print_relative_err)
            txtx = min(x) + (max(x) - min(x))/2.
            txty = min(y) + (max(y) - min(y))/2.

        # make the border be the opposite of the grayscale
        edgecolor = str(-clr_grayscale + 1.)

        # plot
        tile = ax.fill(x, y, color=clr, edgecolor=edgecolor, linewidth=0.1,
            zorder=1)[0]
        # the tiles will be clickable
        if add_clickables:
            # we'll make the tag be the x, y values of tile
            tag = 'x: [%f, %f)\ny: [%f, %f)' %(xlow, xhigh, ylow, yhigh)
            clickable = plot_utils.ClickableElement(tile, 'poly',
                data=this_cube, tag=tag, link=this_cube.html_page)
            ax.get_figure().add_clickable(clickable) 
        if annotate:
            pts = ax.annotate(txt_str, (txtx, txty), ha='center',
                va='center', color=txt_clr, zorder=3, fontsize=fontsize)
    if add_colorbar:
        sm = pyplot.cm.ScalarMappable(cmap=colormap,
            norm=pyplot.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array(zvals)
        if logz and cbformat is None:
            cbformat = pyplot.FuncFormatter(plot_utils.ColorBarLog10Formatter)
        cb = pyplot.colorbar(sm, format=cbformat)
        # if vmin (vmax) is > (<) the min (max) z value, use a bounded
        # formatter
        if vmin > zvals.min() or vmax < zvals.max() and cbformat is None:
            bounded_formatter = plot_utils.create_bounded_colorbar_formatter(
                vmin, vmax, formatter=cb.formatter)
            cb.formatter = bounded_formatter
            cb.update_ticks()
    else:
        cb = None

    # set plot parameters
    plot_xmin, plot_xmax = ax.get_xlim()
    plot_ymin, plot_ymax = ax.get_ylim()
    if xmin is not None:
        plot_xmin = xmin
        if logx:
            plot_xmin = numpy.log10(plot_xmin)
    if xmax is not None:
        plot_xmax = xmax
        if logx:
            plot_xmax = numpy.log10(plot_xmax)
    if ymin is not None:
        plot_ymin = ymin
        if logy:
            plot_ymin = numpy.log10(plot_ymin)
    if ymax is not None:
        plot_ymax = ymax
        if logy:
            plot_ymax = numpy.log10(plot_ymax)

    # have to do log scales manually
    if logx:
        # make the limits slightly larger; to get the ticks right; we'll
        # fix it to be correct later on
        ax.set_xlim(plot_xmin - 1., plot_xmax + 1.)
        new_ticks, ticklabels = get_log10_ticks(ax.get_xticks())
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(ticklabels)
    if logy:
        ax.set_ylim(plot_ymin - 1., plot_ymax + 1.)
        new_ticks, ticklabels = get_log10_ticks(ax.get_yticks())
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(ticklabels)

    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)

    if xlabel is not None:
        # make room from x-label
        orig_pos = ax.get_position()
        new_y0 = 0.15
        add_amount = new_y0 - orig_pos.y0
        new_height = orig_pos.height - add_amount
        ax.set_position([orig_pos.x0, new_y0, orig_pos.width, new_height])
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # set axis color schemes for black background
    ax.tick_params(axis='x', color='w')
    ax.tick_params(axis='y', color='w')
    for side in ['top', 'left', 'right', 'bottom']:
        ax.spines[side].set_color('w')

    return ax, cb


def _adjust_lims_for_subtiles(master_ax, phyper_cubes, xarg, yarg, logx=False,
    logy=False):
    """
    Calculates how the x and y limits of a tiles plot should be adjusted
    to make room for x and y labels on a the lowest-left tile (for subtiles
    plotting). Also returns the index of the lowest-left tile in the list
    of phyper_cubes.
    """
    mfig = master_ax.get_figure()
    master_xlims = master_ax.get_xlim()
    master_ylims = master_ax.get_ylim()
    # we'll want to know which tile is in the furthest lower-left of the plot
    # to determine which inset axis to put labels on
    tile_xvals = numpy.array([parent.get_bound(xarg)[0] \
        for parent in phyper_cubes])
    if logx:
        tile_xvals = numpy.log10(tile_xvals)
    tile_yvals = numpy.array([parent.get_bound(yarg)[0] \
        for parent in phyper_cubes])
    if logy:
        tile_yvals = numpy.log10(tile_yvals)
    label_idx = numpy.intersect1d(
        numpy.where(tile_xvals == tile_xvals.min())[0],
        numpy.where(tile_yvals == tile_yvals.min())[0])[0]
    # the transformation to go from display coordinates to figure coordinates;
    # we'll need this to properly place the inset axes
    invtrans = mfig.transFigure.inverted()
    # we also need to adjust the x and y limits of the master plot 
    # to make room for the inset axes labels
    masterax_lbcoords_fig = numpy.array([master_ax.get_position().xmin,
        master_ax.get_position().ymin])
    tile_leftbottom_coords = numpy.array([tile_xvals.min(), tile_yvals.min()])
    tile_lbcoords_fig = invtrans.transform(
        master_ax.transData.transform(tile_leftbottom_coords))
    axes_dx = tile_lbcoords_fig[0] - masterax_lbcoords_fig[0]
    axes_dy = tile_lbcoords_fig[1] - masterax_lbcoords_fig[1]
    # The following numbers came from trial and error with dpi=300, fontsize=8;
    # they may not work for different dpi and fontsizes
    xlabel_height = 0.085
    ylabel_width = 0.06
    transx = 0
    if axes_dx < ylabel_width:
        # get where the left edge of the tile needs to be in figure coords
        transx = ylabel_width - axes_dx
    transy = 0
    if axes_dy < xlabel_height:
        transy = xlabel_height - axes_dy
    if transx != 0 or transy != 0:
        new_tile_coords = tile_lbcoords_fig + numpy.array([transx, transy])
        # convert to data coordinates: first, transform from fig to display
        new_tile_coords = mfig.transFigure.transform(new_tile_coords)
        # now from display to data coords
        new_tile_coords = master_ax.transData.inverted().transform(
            new_tile_coords)
        # the difference between the new_tile_coords and the old tile coords
        # is how much we need to shift the lower limit of x and y back by
        padx, pady = new_tile_coords - tile_leftbottom_coords
        master_xlims = (master_xlims[0]-padx, master_xlims[1])
        master_ylims = (master_ylims[0]-pady, master_ylims[1])

    return master_xlims, master_ylims, label_idx
    


#----------------------------------------------------------------------
#
#               Volume vs. ranking statistic plots
#
#----------------------------------------------------------------------

def plot_volume_vs_stat_on_axes(ax, phyper_cube, min_stat, max_stat,
        logx=False, logy=False, nbins=20, threshold=None, color='b',
        livetime=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Creats a plot of sensitive volume versus ranking stat on the given axes.
    If a livetime is provided (in s), VT is plotted (with T in Myr) instead
    of just V.
    """
    if phyper_cube.nsamples < 2:
        return plot_utils.empty_plot(ax), None, None
    if logx:
        thresholds = numpy.logspace(numpy.log10(min_stat),
            numpy.log10(max_stat), nbins)
    else:
        thresholds = numpy.linspace(min_stat, max_stat, nbins)
    # volumes is a 2D array: the first column is the volumes, the
    # second the error. If old_method is used, the third column is the
    # lower-bound of the volume.
    volumes = numpy.array([phyper_cube.get_volume(xhat) for xhat in \
        thresholds])
    Vs = volumes[:,0]
    errs_high = volumes[:,1]
    if phyper_cube.use_distance_bins:
        errs_low = volumes[:,2]
    else:
        errs_low = errs_high
    if livetime is not None:
        livetime = 1e-6 * livetime / lal.YRJUL_SI
        Vs *= livetime
        errs_low = errs_low * livetime
        errs_high = errs_high * livetime
    vline, = ax.plot(thresholds, Vs, color=color, lw=2, zorder=2)
    # we'll plot the error region a filled space; fill works counter-clockwise
    # around the polygon formed by the error region
    xvals = numpy.zeros(2*Vs.shape[0])
    xvals[:Vs.shape[0]] = thresholds
    xvals[Vs.shape[0]:] = thresholds[::-1]
    yvals = numpy.zeros(2*Vs.shape[0])
    yvals[:Vs.shape[0]] = Vs - errs_low
    yvals[Vs.shape[0]:] = (Vs + errs_high)[::-1]
    # if logy, replace any negative or 0 values with the smallest non-zero
    if logy and yvals.min() <= 0:
        replace_idx = numpy.where(yvals[:Vs.shape[0]] <= 0.)
        if ymin is None:
            ok_vals = yvals[numpy.where(yvals[:Vs.shape[0]] > 0.)]
            if len(ok_vals) == 0:
                replace_val = Vs.min()*0.01
            else:
                replace_val = ok_vals.min() * 0.1
            ymin = replace_val.min()
        else:
            replace_val = ymin
        yvals[replace_idx] = replace_val
    err_region = ax.fill(xvals, yvals, facecolor=color, lw=0,
        alpha=0.4, zorder=1)
    if logx and logy:
        ax.loglog()
    elif logx:
        ax.semilogx()
    elif logy:
        ax.semilogy()
    plot_xmin, plot_xmax = ax.get_xlim()
    plot_ymin, plot_ymax = ax.get_ylim()
    if xmin is not None:
        plot_xmin = xmin
    if xmax is not None:
        plot_xmax = xmax
    if ymin is not None:
        plot_ymin = ymin
    if ymax is not None:
        plot_ymax = ymax
    # if a threshold is given, plot a line showing where it is
    if threshold is not None:
        tline = ax.plot([threshold, threshold], [plot_ymin, plot_ymax],
            'r--', lw=2, zorder=3)
    else:
        tline = None
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)

    return xvals, yvals, vline


def plot_volume_vs_stat(phyper_cube, min_stat, max_stat, stat_label,
        logx=False, logy=False, nbins=20, threshold=None, color='b',
        livetime=None, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Creates a plot of sensitive volume vs stat.
    """
    # we'll use a mappable figure even though this has no clickable elements
    fig = plot_utils.figure()
    fig.subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(111)

    # add the plot to axes
    plot_volume_vs_stat_on_axes(ax, phyper_cube, min_stat, max_stat,
        logx=logx, logy=logy, nbins=nbins, threshold=threshold,
        livetime=livetime,
        color=color, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    # label
    ax.set_xlabel(stat_label)
    if livetime is not None:
        ax.set_ylabel(r'$VT\,(\mathrm{Mpc}^3\,\mathrm{Myr})$')
    else:
        ax.set_ylabel(r'$\mathcal{V}\,(\mathrm{Mpc}^3)$')

    return fig

def plot_multivolumes_vs_stat(phyper_cubes, min_stat, max_stat, stat_label,
        colors, labels, livetimes=[], logx=False, logy=False, nbins=20,
        threshold=None,
        xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Plots the volume computed from each phyper cube in the given list of
    phyper cubes as a separate line on the axis.
    """
    fig = plot_utils.figure(figsize=(5,4))
    fig.subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(111)

    # add each plot to the axes
    lines = []
    for ii,(pcube,color) in enumerate(zip(phyper_cubes, colors)):
        # if a livetime is specified, get it
        if livetimes != []:
            livetime = livetimes[ii]
        else:
            livetime = None
        if color == '' or color is None:
            if len(phyper_cubes) == 1:
                clrval = 0. 
            else:
                clrval = ii/float(len(phyper_cubes)-1)
            color = pyplot.cm.jet_r(clrval)
        _, _, line = plot_volume_vs_stat_on_axes(ax, pcube, min_stat, max_stat,
            logx=logx, logy=logy, nbins=nbins, threshold=threshold,
            color=color, livetime=livetime, xmin=xmin, xmax=xmax, ymin=ymin,
            ymax=ymax)
        lines.append(line)
    # label
    ax.legend(lines, labels)
    ax.set_xlabel(stat_label)
    if livetimes != []:
        ax.set_ylabel('$VT\,(\mathrm{Mpc}^3\,\mathrm{Myr})$')
    else:
        ax.set_ylabel('$\mathcal{V}\,(\mathrm{Mpc}^3)$')

    return fig

def plot_twovolume_vs_stat(phyper_cube, test_min_stat, test_max_stat,
        ref_min_stat, ref_max_stat, test_threshold, ref_threshold,
        test_label='test', ref_label='reference',
        test_livetime=None, ref_livetime=None,
        logx=False, logy=False, nbins=100, test_color=pyplot.cm.jet(1.),
        ref_color=pyplot.cm.jet(0.),
        test_xmin=None, test_xmax=None, ref_xmin=None, ref_xmax=None,
        ymin=None, ymax=None):
    """
    Creates a plot of the sensitive volume of a test set of results and
    a reference set of results on the same axis. The two x-axes are aligned
    such that the test and reference thresholds line up.
    """
    # we'll use a mappable figure even though this has no clickable elements
    fig = plot_utils.figure()
    fig.subplots_adjust(bottom=0.15)
    ax = fig.add_subplot(111)

    if (ref_livetime is None and test_livetime is not None) or \
            (ref_livetime is not None and test_livetime is None):
        raise ValueError('must provide a livetime for both reference and test')

    # plot the reference volume using the bottom x-axis
    # Note: we'll set the threshold to None; we'll plot it after both
    # reference and test have been plotted
    _, yvals, refline = plot_volume_vs_stat_on_axes(ax,
        phyper_cube.reference_cube,
        ref_min_stat, ref_max_stat, logx=logx, logy=logy, nbins=nbins,
        threshold=None, livetime=ref_livetime, color=ref_color,
        xmin=ref_xmin, xmax=ref_xmax,
        ymin=ymin, ymax=ymax)
    # label
    ax.set_xlabel(phyper_cube.reference_cube.stat_label)
    plot_ymin, plot_ymax = yvals.min(), yvals.max()

    # now create a separate axis that shares the same x-axis
    # we'll only do this if there is a difference between the two axes
    use_single_ax = test_min_stat == ref_min_stat and \
        test_max_stat == ref_max_stat and test_threshold == ref_threshold
    if use_single_ax:
        # just use a single axis
        ax2 = ax
    else:
        ax2 = ax.twiny()
    _, yvals, testline = plot_volume_vs_stat_on_axes(ax2,
        phyper_cube.test_cube,
        test_min_stat, test_max_stat, logx=logx, logy=False, nbins=nbins,
        threshold=None, livetime=test_livetime, color=test_color,
        xmin=test_xmin, xmax=test_xmax, ymin=ymin, ymax=ymax)
    ax2.set_xlabel(phyper_cube.test_cube.stat_label)
    plot_ymin, plot_ymax = min(plot_ymin, yvals.min()), \
        max(plot_ymax, yvals.max())
    
    # figure out limits
    if ymin is None:
        if logy:
            ymin = plot_ymin*10**(-0.5)
        else:
            ymin = 0.9*plot_ymin
    if ymax is None:
        if logy:
            ymax = plot_ymax*10**(0.5)
        else:
            ymax = 1.1*plot_ymax
    # now align the two axes such that the thresholds are equal; we'll do
    # this by adjusting the test xlims
    if not use_single_ax:
        test_xmin, test_xmax = ax2.get_xlim()
        # find the displacement from the test threshold to the reference
        # threshold in units of the test x-axis; first, we'll get the location
        # of the reference threshold in display units
        ref_thresh_coords = ax.transData.transform_point([ref_threshold,
            (ymax-ymin)/2.]) 
        # now convert that to test's x-axis units
        ref_thresh_loc = ax2.transData.inverted().transform_point(
            ref_thresh_coords)
        dx = ref_thresh_loc[0] - test_threshold
        ax2.set_xlim(test_xmin-dx, test_xmax-dx)

    # now plot the thershold line
    ax.plot([ref_threshold, ref_threshold], [ymin, ymax],
        'r--', lw=2, zorder=3)
    ax.set_ylim(ymin, ymax)

    if ref_livetime is not None:
        ax.set_ylabel(r'$VT\,(\mathrm{Mpc}^3\,\mathrm{Myr})$')
    else:
        ax.set_ylabel(r'$\mathcal{V}\,(\mathrm{Mpc}^3)$')

    # create the legend
    ax.legend([testline, refline], ['%s'%(test_label), '%s'%(ref_label)])

    return fig


def plot_volume_vs_stat_from_layer(layer, min_stat, max_stat, stat_label,
        include_children=False, user_tag='', min_ninj=2,
        logx=False, logy=False, nbins=20, threshold=None, livetime=None,
        color='b',
        xmin=None, xmax=None, ymin=None, ymax=None, dpi=300, verbose=False):
    """
    Wrapper around plot_volume_vs_stat that creates a volume_vs_stat plot for
    every parent in the given layer. The resulting figures are saved to disk,
    with the MappableFigure instances saved to each parent's
    volume_vs_stat_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_volume_vs_stat{-user_tag}-{layer_level}-\
    # {file_index}.png
    fnametmplt = '%s/plot_volume_vs_stat%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    plot_cubes = [parent for parent in layer.parents \
        if parent.nsamples >= min_ninj]
    # if include_children, we'll also create plots for the children
    if include_children:
        plot_cubes += [child for parent in plot_cubes \
            for child in parent.children if child.nsamples >= min_ninj]
    if verbose:
        print >> sys.stdout, "creating volume vs. stat plots for level %i:" %(
            layer.level)
    for ii,cube in enumerate(plot_cubes):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(plot_cubes)),
            sys.stdout.flush()
        fig = plot_volume_vs_stat(cube, min_stat, max_stat, stat_label,
            logx=logx, logy=logy, nbins=nbins, threshold=threshold,
            livetime=livetime,
            color=color, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        fig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        cube.volumes_vs_stat_plot = fig
    if verbose:
        print >> sys.stdout, ""


def plot_twovolume_vs_stat_from_layer(layer, test_min_stat, test_max_stat,
        ref_min_stat, ref_max_stat, test_threshold, ref_threshold,
        test_livetime=None, ref_livetime=None,
        test_label='test', ref_label='reference',
        include_children=False, user_tag='', min_ninj=2,
        logx=False, logy=False, nbins=100, test_color=pyplot.cm.jet(1.),
        ref_color=pyplot.cm.jet(0.), livetime=None,
        test_xmin=None, test_xmax=None, ymin=None, ymax=None, dpi=300,
        verbose=False):
    """
    Wrapper around plot_twovolume_vs_stat that creates a twovolume_vs_stat plot
    for every parent in the given layer. The resulting figures are saved to
    disk, with the MappableFigure instances saved to each parent's
    volume_vs_stat_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_twovolume_vs_stat{-user_tag}-{layer_level}-\
    # {file_index}.png
    fnametmplt = '%s/plot_twovolume_vs_stat%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    plot_cubes = [parent for parent in layer.parents \
        if parent.nsamples >= min_ninj]
    # if include_children, we'll also create plots for the children
    if include_children:
        plot_cubes += [child for parent in plot_cubes \
            for child in parent.children if child.nsamples >= min_ninj]
    if verbose:
        print >> sys.stdout, "creating volume vs. stat plots for level %i:" %(
            layer.level)
    for ii,cube in enumerate(plot_cubes):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(plot_cubes)),
            sys.stdout.flush()
        fig = plot_twovolume_vs_stat(cube, test_min_stat, test_max_stat,
            ref_min_stat, ref_max_stat, test_threshold, ref_threshold,
            test_label=test_label, ref_label=ref_label,
            test_livetime=test_livetime, ref_livetime=ref_livetime,
            logx=logx, logy=logy, nbins=nbins, test_color=test_color,
            ref_color=ref_color, test_xmin=test_xmin, test_xmax=test_xmax,
            ymin=ymin, ymax=ymax)

        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        fig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        cube.volumes_vs_stat_plot = fig
    if verbose:
        print >> sys.stdout, ""


#----------------------------------------------------------------------
#
#               Volume tile plots
#
#----------------------------------------------------------------------

def plot_volumes(phyper_cubes, xarg, xlabel, yarg, ylabel, threshold,
        livetime=None, min_ninj=2, tmplt_label='', inj_label='',
        add_title=True,
        colormap='hot', maxvol=None, minvol=None, add_colorbar=False,
        annotate=True, fontsize=8, print_relative_err=False, 
        logx=False, logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, fig=None,
        ax=None, add_clickables=True, dpi=300):
    """
    Given a list of phyper_cubes, creates a tile plot of the volumes.
    """

    if fig is None:
        mfig = plot_utils.figure(dpi=dpi)
    else:
        mfig = fig
    if ax is None:
        ax = mfig.add_subplot(111, axisbg='k')
    else:
        # check that the given axes is in fig
        if ax not in mfig.axes:
            raise ValueError("given ax is not in the figure")

    # only use cubes that have enough injections
    phyper_cubes = [this_cube for this_cube in phyper_cubes if \
        this_cube.nsamples >= min_ninj and \
        this_cube.get_volume(threshold)[0] != 0.]

    # if there is nothing to plot, just create an empty plot and return
    if len(phyper_cubes) == 0:
        plot_utils.empty_plot(ax)
        return mfig

    # volumes is a 2D array; the first column are the volumes, the
    # second the plus error, the third the minus error
    volumes = numpy.array([this_cube.get_volume(threshold) \
        for this_cube in phyper_cubes]).astype(numpy.float)
    Vs = volumes[:,0]
    plus_errs = volumes[:,1]
    minus_errs = volumes[:,2]
    if livetime is not None:
        livetime = 1e-6 * livetime / lal.YRJUL_SI
        Vs *= livetime
        plus_errs = plus_errs * livetime
        minus_errs = minus_errs * livetime
    if minvol is None:
        minvol = Vs[numpy.nonzero(Vs)].min()
    if maxvol is None:
        maxvol = Vs.max()
    # we'll divide by the closest power of 10 of the smallest value of these
    # volumes
    if livetime is None:
        conversion_factor = numpy.floor(numpy.log10(minvol))
        Vs *= 10**(-conversion_factor)
        plus_errs *= 10**(-conversion_factor)
        minus_errs *= 10**(-conversion_factor)
        minvol *= 10**(-conversion_factor)
        maxvol *= 10**(-conversion_factor)
        # if the conversion factor is > 10^9, we'll change the label to Gpc
        # (V's are assumed to be in Mpc^3)
        if conversion_factor >= 9:
            conversion_factor -= 9.
            units = 'Gpc'
        else:
            units = 'Mpc'
        if conversion_factor == 0.:
            prefactor = ''
        elif conversion_factor == 1.:
            prefactor = '10'
        else:
            prefactor = '10^{%i}' %(int(conversion_factor))
        units_label = '%s\,\mathrm{%s}^3' %(prefactor, units)
    else:
        # if livetime is specified, we'll always plot Mpc^3 Myr
        units_label = '\mathrm{Mpc}^3\,\mathrm{Myr}'

    # create a tiles plot on the axis
    _, cb = _plot_tiles(ax, Vs, plus_errs, minus_errs, phyper_cubes,
        xarg, xlabel, yarg, ylabel,
        colormap=colormap, vmax=maxvol, vmin=minvol, add_colorbar=add_colorbar,
        annotate=annotate, fontsize=fontsize,
        print_relative_err=print_relative_err, logx=logx, logy=logy, logz=logz,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        add_clickables=add_clickables)

    if add_title:
        if livetime is None:
            title = r'$\mathcal{V}_{%s%s}~(%s)$' %(tmplt_label, inj_label,
                units_label)
        else:
            title = r'$V_{%s%s}T\,(%s)$' %(tmplt_label, inj_label,
                units_label)
        # if we are adding a colorbar, put the title on its axis
        if add_colorbar:
            cb.ax.set_ylabel(title)
        else:
            ax.set_title(title)

    return mfig


def plot_subvolumes(phyper_cubes, xarg, xlabel, yarg, ylabel,
        sub_xarg, sub_xlabel, sub_yarg, sub_ylabel, threshold, 
        livetime=None, min_ninj=2, tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None,
        logx=False, logy=False, sub_logx=False, sub_logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, dpi=300):
    """
    Similar to plot_volumes, but each tile shows the volumes of a sub-layer
    of tiles.
    """
    # only use phyper_cubes that have enough injections
    phyper_cubes = [parent for parent in phyper_cubes \
        if parent.nsamples >= min_ninj and \
        parent.get_volume(threshold)[0] != 0.]

    # if nothing to plot, just create an empty plot and return
    if phyper_cubes == []:
        mfig = plot_utils.figure(dpi=dpi)
        ax = mfig.add_subplot(111)
        plot_utils.empty_plot(ax)
        return mfig

    # to ensure we get properly normalized colors, find the largest
    # and smallest Vs across all of the children of all of the phyper_cubes
    if minvol is None or maxvol is None:
        Vs = numpy.array([
            child.get_volume(threshold) for parent in phyper_cubes \
            for child in parent.children if child.nsamples >= min_ninj and \
            child.get_volume(threshold)[0] != 0.])[:,0]
        if livetime is not None:
            Vs *= 1e-6 * livetime / lal.YRJUL_SI
        if minvol is None:
            minvol = Vs.min()
        if maxvol is None:
            maxvol = Vs.max()

    # create the master plot with clickable elements
    mfig = plot_volumes(
        phyper_cubes, xarg, xlabel, yarg, ylabel, threshold, min_ninj=min_ninj,
        livetime=livetime,
        tmplt_label=tmplt_label, inj_label=inj_label,
        add_title=True, colormap=colormap, maxvol=maxvol, minvol=minvol,
        add_colorbar=True, annotate=False, logx=logx, logy=logy, logz=logz,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, dpi=dpi,
        add_clickables=True)
      
    master_ax = mfig.axes[0]
    
    # get the x and y limits we need in order to have enough room for the
    # sub-tiles labels, along with the index of the sub tile to label
    master_xlims, master_ylims, label_idx = _adjust_lims_for_subtiles(
        master_ax, phyper_cubes, xarg, yarg, logx=logx, logy=logy)
    master_ax.set_xlim(master_xlims)
    master_ax.set_ylim(master_ylims)
        
    # cycle over phyper_cubes that were plotted, creating sub-plots from their
    # children
    invtrans = mfig.transFigure.inverted()
    for ii,clickable in enumerate(mfig.clickable_elements):
        tile = clickable.element
        parent = clickable.data
        # create an axis to plot this family on; this axis will cover the tile
        # exactly
        tile_fig_coords = invtrans.transform(tile.axes.transData.transform(
            tile.get_xy()))
        # tile fig coords are a 2D array, in which the first column is the
        # x values of the vertices of the tile, the second column are the y;
        # we need [left, bottom, width, height] for the inset axes
        tile_figx = tile_fig_coords[:,0]
        tile_figy = tile_fig_coords[:,1]
        inset_coords = [tile_figx.min(), tile_figy.min(),
            tile_figx.max()-tile_figx.min(), tile_figy.max()-tile_figy.min()]
        inset_axes = mfig.add_axes(inset_coords, axisbg='k')
        # now create the sub plot in the inset_axes
        # note that we turn off the clickables in the inset_axes; also, we'll
        # add the axis labels later
        # if there are not enough injections in all of the tiles, just create
        # a hatched empty plot
        if not numpy.array([child.nsamples >= min_ninj \
            for child in parent.children]).any():
                plot_utils.empty_hatched_plot(inset_axes, hatch="x") 
        else:
            plot_volumes(
                parent.children, sub_xarg, None, sub_yarg, None, threshold,
                livetime=livetime, min_ninj=min_ninj, add_title=False,
                colormap=colormap,
                maxvol=maxvol, minvol=minvol,
                add_colorbar=False, annotate=False,
                logx=sub_logx, logy=sub_logy, logz=logz,
                fig=mfig, ax=inset_axes, add_clickables=False)
            sub_xmin = min([child.get_bound(sub_xarg)[0] \
                for child in parent.children])
            sub_xmax = max([child.get_bound(sub_xarg)[1] \
                for child in parent.children])
            if sub_logx:
                sub_xmin = numpy.log10(sub_xmin)
                sub_xmax = numpy.log10(sub_xmax)
            sub_ymin = min([child.get_bound(sub_yarg)[0] \
                for child in parent.children])
            sub_ymax = max([child.get_bound(sub_yarg)[1] \
                for child in parent.children])
            if sub_logy:
                sub_ymin = numpy.log10(sub_ymin)
                sub_ymax = numpy.log10(sub_ymax)
            inset_axes.set_xlim(sub_xmin, sub_xmax)
            inset_axes.set_ylim(sub_ymin, sub_ymax)
        if ii == label_idx:
            # make the ticklabels white
            if sub_logx: 
                _, ticklabels = get_log10_ticks(inset_axes.get_xticks())
            else:
                ticklabels = map(str, inset_axes.get_xticks())
            inset_axes.set_xticklabels(ticklabels, color='w', fontsize=6)
            if sub_logy: 
                _, ticklabels = get_log10_ticks(inset_axes.get_yticks())
            else:
                ticklabels = map(str, inset_axes.get_yticks())
            inset_axes.set_yticklabels(ticklabels, color='w', fontsize=6)
            # add the labels
            inset_axes.set_xlabel(sub_xlabel, color='w',
                fontsize=8, labelpad=2)
            inset_axes.set_ylabel(sub_ylabel, color='w',
                fontsize=8, labelpad=2)
        else:
            # remove tick and axis labels otherwise
            inset_axes.set_xticklabels([])
            inset_axes.set_yticklabels([])
            inset_axes.set_xlabel('')
            inset_axes.set_ylabel('')

    return mfig


#----------------------------------------------------------------------
#
#               Relative gain tile plots
#
#----------------------------------------------------------------------

def plot_gains(phyper_cubes, xarg, xlabel, yarg, ylabel, test_threshold,
        ref_threshold, test_livetime=None, ref_livetime=None, min_ninj=2,
        test_label='', ref_label='',
        add_title=True, colormap='RdBu_r', maxgain=None, mingain=None,
        add_colorbar=False, cbformat=None, annotate=True, fontsize=8,
        print_relative_err=False, logx=False, logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, fig=None,
        ax=None, add_clickables=True, dpi=300):
    """
    Given a list of PHyperCubeGains, creates a tile plots of the fractional
    gains.
    """
    if (ref_livetime is None and test_livetime is not None) or \
            (ref_livetime is not None and test_livetime is None):
        raise ValueError('must provide a livetime for both reference and test')
    if ref_livetime is not None:
        time_factor = float(test_livetime) / ref_livetime
    else:
        time_factor = 1.
    if fig is None:
        mfig = plot_utils.figure(dpi=dpi)
    else:
        mfig = fig
    if ax is None:
        ax = mfig.add_subplot(111, axisbg='k')
    else:
        # check that the given axes is in fig
        if ax not in mfig.axes:
            raise ValueError("given ax is not in the figure")

    # only use cubes that have enough injections
    phyper_cubes = [this_cube for this_cube in phyper_cubes if \
        this_cube.nsamples >= min_ninj and \
        numpy.isfinite(this_cube.get_fractional_gain(
            ref_threshold, test_threshold)[0])]

    # if there is nothing to plot, just create an empty plot and return
    if len(phyper_cubes) == 0:
        plot_utils.empty_plot(ax)
        return mfig

    # gains is a 2D array; the first column are the gains, the
    # second the error
    gains = numpy.array([this_cube.get_fractional_gain(ref_threshold,
        test_threshold) \
        for this_cube in phyper_cubes]).astype(numpy.float)
    Gs = time_factor * gains[:,0]
    gain_errs = time_factor * gains[:,1]

    # if mingain and maxgain are not provided, and coloring as logz, make the
    # min/max symmetric about 1
    if mingain is None and maxgain is None and logz:
        pwr = abs(numpy.log10(Gs)).max()
        # we'll make the spread be atleast +/-10%
        if pwr < 0.04:
            pwr = 0.04 
        mingain = 10**(-pwr)
        maxgain = 10**(pwr)
    # otherwise, just use the min/max
    if mingain is None:
        mingain = Gs.min()
    if maxgain is None:
        maxgain = Gs.max()

    # create a tiles plot on the axis
    _, cb = _plot_tiles(ax, Gs, gain_errs, gain_errs, phyper_cubes,
        xarg, xlabel, yarg, ylabel,
        colormap=colormap, vmax=maxgain, vmin=mingain,
        add_colorbar=add_colorbar, cbformat=cbformat,
        annotate=annotate, fontsize=fontsize,
        print_relative_err=print_relative_err, logx=logx, logy=logy, logz=logz,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        add_clickables=add_clickables)

    if add_title:
        if ref_livetime is not None:
            Glbl = r'$\mathcal{G}_{VT}$'
        else:
            Glbl = r'$\mathcal{G}$'
        title = '%s / %s' %(test_label, ref_label)
        if add_colorbar:
            cb.ax.set_ylabel(Glbl, rotation=0, labelpad=-2)
        else:
            title = '%s (%s)' %(Glbl, title)
        ax.set_title(title)

    return mfig


def plot_subgains(phyper_cubes, xarg, xlabel, yarg, ylabel,
        sub_xarg, sub_xlabel, sub_yarg, sub_ylabel, test_threshold,
        ref_threshold, test_livetime=None, ref_livetime=None, min_ninj=2,
        test_label='', ref_label='',
        colormap='RdBu_r',  maxgain=None, mingain=None,
        logx=False, logy=False, sub_logx=False, sub_logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, dpi=300):
    """
    Similar to plot_gains, but each tile shows the relative gains of a
    sub-layer of tiles.
    """
    if (ref_livetime is None and test_livetime is not None) or \
            (ref_livetime is not None and test_livetime is None):
        raise ValueError('must provide a livetime for both reference and test')
    if ref_livetime is not None:
        time_factor = float(test_livetime) / ref_livetime
    else:
        time_factor = 1.

    # only use phyper_cubes that have enough injections
    phyper_cubes = [this_cube for this_cube in phyper_cubes if \
        this_cube.nsamples >= min_ninj and \
        numpy.isfinite(this_cube.get_fractional_gain(
            ref_threshold, test_threshold)[0])]

    # if nothing to plot, just create an empty plot and return
    if phyper_cubes == []:
        mfig = plot_utils.figure(dpi=dpi)
        ax = mfig.add_subplot(111)
        plot_utils.empty_plot(ax)
        return mfig

    # to ensure we get properly normalized colors, find the largest
    # and smallest gains across all of the children of all of the phyper_cubes
    Gs = time_factor * numpy.array([
        child.get_fractional_gain(ref_threshold, test_threshold)
        for parent in phyper_cubes \
        for child in parent.children if child.nsamples >= min_ninj and \
            numpy.isfinite(child.get_fractional_gain(ref_threshold,
            test_threshold)[0])])[:,0]
    # if mingain and maxgain are not provided, and coloring as logz, make the
    # min/max symmetric about 1
    if mingain is None and maxgain is None and logz:
        pwr = abs(numpy.log10(Gs)).max()
        # we'll make the spread be atleast +/-10%
        if pwr < 0.04:
            pwr = 0.04 
        mingain = 10**(-pwr)
        maxgain = 10**(pwr)
    # otherwise, just use the min/max
    if mingain is None:
        mingain = Gs.min()
    if maxgain is None:
        maxgain = Gs.max()

    # if mingain (maxgain) is > (<) the min (max) gain of the subtiles, set
    # the colorbar formatter to be a bounded formatter;
    # we need to do this here because the
    # call to plot_gains won't know about the sub-tiles' gains
    if mingain > Gs.min() or maxgain < Gs.max():
        cbformat = plot_utils.create_bounded_colorbar_formatter(
            mingain, maxgain,
            formatter=plot_utils.ColorBarLog10Formatter if logz else None)
    else:
        cbformat = None

    # create the master plot with clickable elements
    mfig = plot_gains(phyper_cubes, xarg, xlabel, yarg, ylabel, test_threshold,
        ref_threshold, min_ninj=min_ninj,
        test_livetime=test_livetime, ref_livetime=ref_livetime,
        test_label=test_label, ref_label=ref_label,
        add_title=True, cbformat=cbformat, colormap=colormap,
        maxgain=maxgain, mingain=mingain,
        add_colorbar=True, annotate=False, logx=logx, logy=logy, logz=logz,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, add_clickables=True,
        dpi=dpi)
      
    master_ax = mfig.axes[0]
    
    # get the x and y limits we need in order to have enough room for the
    # sub-tiles labels, along with the index of the sub tile to label
    master_xlims, master_ylims, label_idx = _adjust_lims_for_subtiles(
        master_ax, phyper_cubes, xarg, yarg, logx=logx, logy=logy)
    master_ax.set_xlim(master_xlims)
    master_ax.set_ylim(master_ylims)
        
    # cycle over phyper_cubes that were plotted, creating sub-plots from their
    # children
    invtrans = mfig.transFigure.inverted()
    for ii,clickable in enumerate(mfig.clickable_elements):
        tile = clickable.element
        parent = clickable.data
        # create an axis to plot this family on; this axis will cover the tile
        # exactly
        tile_fig_coords = invtrans.transform(tile.axes.transData.transform(
            tile.get_xy()))
        # tile fig coords are a 2D array, in which the first column is the
        # x values of the vertices of the tile, the second column are the y;
        # we need [left, bottom, width, height] for the inset axes
        tile_figx = tile_fig_coords[:,0]
        tile_figy = tile_fig_coords[:,1]
        inset_coords = [tile_figx.min(), tile_figy.min(),
            tile_figx.max()-tile_figx.min(), tile_figy.max()-tile_figy.min()]
        inset_axes = mfig.add_axes(inset_coords, axisbg='k')
        # now create the sub plot in the inset_axes
        # note that we turn off the clickables in the inset_axes; also, we'll
        # add the axis labels later
        # if there are not enough injections in all of the tiles, just create
        # a hatched empty plot
        if not numpy.array([child.nsamples >= min_ninj \
            for child in parent.children]).any():
                plot_utils.empty_hatched_plot(inset_axes, hatch="x") 
        else:
            plot_gains(
                parent.children, sub_xarg, None, sub_yarg, None,
                test_threshold, ref_threshold, min_ninj=min_ninj,
                test_livetime=test_livetime, ref_livetime=ref_livetime,
                add_title=False, colormap=colormap,
                maxgain=maxgain, mingain=mingain,
                add_colorbar=False, annotate=False,
                logx=sub_logx, logy=sub_logy, logz=logz,
                fig=mfig, ax=inset_axes, add_clickables=False)
            # make the inset axis limits cover exactly the space of the tiles
            sub_xmin = min([child.get_bound(sub_xarg)[0] \
                for child in parent.children])
            sub_xmax = max([child.get_bound(sub_xarg)[1] \
                for child in parent.children])
            if sub_logx:
                sub_xmin = numpy.log10(sub_xmin)
                sub_xmax = numpy.log10(sub_xmax)
            sub_ymin = min([child.get_bound(sub_yarg)[0] \
                for child in parent.children])
            sub_ymax = max([child.get_bound(sub_yarg)[1] \
                for child in parent.children])
            if sub_logy:
                sub_ymin = numpy.log10(sub_ymin)
                sub_ymax = numpy.log10(sub_ymax)
            inset_axes.set_xlim(sub_xmin, sub_xmax)
            inset_axes.set_ylim(sub_ymin, sub_ymax)
        if ii == label_idx:
            # make the ticklabels white
            if sub_logx: 
                _, ticklabels = get_log10_ticks(inset_axes.get_xticks())
            else:
                ticklabels = map(str, inset_axes.get_xticks())
            inset_axes.set_xticklabels(ticklabels, color='w', fontsize=6)
            if sub_logy: 
                _, ticklabels = get_log10_ticks(inset_axes.get_yticks())
            else:
                ticklabels = map(str, inset_axes.get_yticks())
            inset_axes.set_yticklabels(ticklabels, color='w', fontsize=6)
            # add the labels
            inset_axes.set_xlabel(sub_xlabel, color='w',
                fontsize=8, labelpad=2)
            inset_axes.set_ylabel(sub_ylabel, color='w',
                fontsize=8, labelpad=2)
        else:
            # remove tick and axis labels otherwise
            inset_axes.set_xticklabels([])
            inset_axes.set_yticklabels([])
            inset_axes.set_xlabel('')
            inset_axes.set_ylabel('')

    return mfig


#----------------------------------------------------------------------
#
#               Functions to create tile plots from layers
#
#----------------------------------------------------------------------

#
#   Volumes
#
def plot_volumes_from_layer(layer, threshold, livetime=None, user_tag='',
        min_ninj=2, tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None, fontsize=8,
        print_relative_err=False, 
        logz=False, dpi=300, verbose=False):
    """
    Wrapper around plot_volumes that creates a volume plot for every
    parent in the given layer. X and Y arguments/labels of the volume
    plot are retrieved from the given layer's x and y arguments. The resulting
    figures are saved to disk, with the filenames saved to each parent's
    tiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_volumes{-user_tag}-{layer_level}-{parent_index}.png
    fnametmplt = '%s/plot_volumes%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    if verbose:
        print >> sys.stdout, "creating volume tiles plots for level %i:" %(
            layer.level)
    for ii,parent in enumerate(layer.parents):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(layer.parents)),
            sys.stdout.flush()
        mfig = plot_volumes(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label, threshold, livetime=livetime,
            min_ninj=min_ninj,
            tmplt_label=tmplt_label, inj_label=inj_label, annotate=True,
            add_colorbar=False, colormap=colormap, maxvol=maxvol,
            minvol=minvol, fontsize=fontsize,
            print_relative_err=print_relative_err,
            logx=layer.plot_log_x, logy=layer.plot_log_y,
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)
        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        parent.tiles_plot = mfig
    if verbose:
        print >> sys.stdout, ""


def plot_subvolumes_from_layer(layer, threshold, livetime=None, user_tag='',
        min_ninj=2, tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None, logz=False, dpi=300,
        verbose=False):
    """
    Wrapper around plot_subvolumes that creates a subvolume plot for every
    parent in the given layer. X and Y arguments/labels of the volume
    plot are retrieved from the given layer's x and y arguments. X and Y
    arguments/labels for the sub-tiles are retrieved from the given layer's
    sub-layer. The resulting figures are saved to disk, with the filenames
    saved to each parent's subtiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    if not layer.sub_layer:
        raise ValueError('no sub_layer set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_subvolumes{-user_tag}-{layer_level}-{parent_index}
    fnametmplt = '%s/plot_subvolumes%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    if verbose:
        print >> sys.stdout, "creating volume subtiles plots for level %i:" %(
            layer.level)
    for ii,parent in enumerate(layer.parents):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(layer.parents)),
            sys.stdout.flush()
        mfig = plot_subvolumes(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label,
            layer.sub_layer.x_param, layer.sub_layer.x_param.label,
            layer.sub_layer.y_param, layer.sub_layer.y_param.label, threshold,
            livetime=livetime,
            min_ninj=min_ninj, tmplt_label=tmplt_label, inj_label=inj_label,
            colormap=colormap, maxvol=maxvol, minvol=minvol,
            logx=layer.plot_log_x, logy=layer.plot_log_y,
            sub_logx=layer.sub_layer.x_distr == 'log10',
            sub_logy=layer.sub_layer.y_distr == 'log10',
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)

        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        parent.subtiles_plot = mfig

    if verbose:
        print >> sys.stdout, ""


#
#   Relative gains
#
def plot_gains_from_layer(layer, test_threshold, ref_threshold, 
        test_livetime=None, ref_livetime=None, user_tag='',
        min_ninj=2, test_label='', ref_label='',
        colormap='RdBu_r', maxgain=None, mingain=None, fontsize=8,
        print_relative_err=False, logz=False, include_volume_plots=False,
        minvol=None, maxvol=None, dpi=300, verbose=False):
    """
    Wrapper around plot_gains that creates a gain tile plot for every
    parent in the given layer. X and Y arguments/labels of the tile
    plot are retrieved from the given layer's x and y arguments. The resulting
    figures are saved to disk, with the filenames saved to each parent's
    tiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_gains{-user_tag}-{layer_level}-{parent_index}.png
    fnametmplt = '%s/plot_gains%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    if verbose:
        print >> sys.stdout, "creating gain tiles plots for level %i:" %(
            layer.level)
    for ii,parent in enumerate(layer.parents):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(layer.parents)),
            sys.stdout.flush()
        mfig = plot_gains(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label, test_threshold, ref_threshold,
            test_livetime=test_livetime, ref_livetime=ref_livetime,
            min_ninj=min_ninj, test_label=test_label, ref_label=ref_label,
            add_title=True, maxgain=maxgain, mingain=mingain,
            add_colorbar=False, colormap=colormap,
            annotate=True, fontsize=fontsize,
            print_relative_err=print_relative_err,
            logx=layer.plot_log_x, logy=layer.plot_log_y,
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)

        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        parent.tiles_plot = mfig

        # if volume plots are desired, create them
        if include_volume_plots:
            # the test volumes
            mfig = plot_volumes([child.test_cube for child in parent.children],
                layer.x_param, layer.x_param.label,
                layer.y_param, layer.y_param.label, test_threshold,
                livetime=test_livetime,
                min_ninj=min_ninj,
                tmplt_label='', inj_label='', annotate=True,
                add_colorbar=False, colormap='hot', maxvol=maxvol,
                minvol=minvol, fontsize=fontsize,
                print_relative_err=print_relative_err,
                logx=layer.plot_log_x, logy=layer.plot_log_y,
                logz=False, add_clickables=False,
                xmin=layer.plot_x_min, xmax=layer.plot_x_max,
                ymin=layer.plot_y_min, ymax=layer.plot_y_max)
            # save the figure
            plotname = fnametmplt.replace('plot_gains', 'test_volumes') %(
                layer.images_dir, user_tag, layer.level, ii) 
            mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
                dpi=dpi)
            parent.test_cube.tiles_plot = mfig

            # the reference volumes
            mfig = plot_volumes(
                [child.reference_cube for child in parent.children],
                layer.x_param, layer.x_param.label,
                layer.y_param, layer.y_param.label, ref_threshold,
                min_ninj=min_ninj, livetime=ref_livetime,
                tmplt_label='', inj_label='', annotate=True,
                add_colorbar=False, colormap='hot', maxvol=maxvol,
                minvol=minvol, fontsize=fontsize,
                print_relative_err=print_relative_err,
                logx=layer.plot_log_x, logy=layer.plot_log_y,
                logz=True, add_clickables=False,
                xmin=layer.plot_x_min, xmax=layer.plot_x_max,
                ymin=layer.plot_y_min, ymax=layer.plot_y_max)
            # save the figure
            plotname = fnametmplt.replace('plot_gains', 'ref_volumes') %(
                layer.images_dir, user_tag, layer.level, ii) 
            mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
                dpi=dpi)
            parent.reference_cube.tiles_plot = mfig

    if verbose:
        print >> sys.stdout, ""


def plot_subgains_from_layer(layer, test_threshold, ref_threshold, user_tag='',
        min_ninj=2, test_livetime=None, ref_livetime=None,
        test_label='', ref_label='',
        colormap='RdBu_r', maxgain=None, mingain=None, logz=False, dpi=300,
        verbose=False):
    """
    Wrapper around plot_subgains that creates a subgain plot for every
    parent in the given layer. X and Y arguments/labels of the gain
    plot are retrieved from the given layer's x and y arguments. X and Y
    arguments/labels for the sub-tiles are retrieved from the given layer's
    sub-layer. The resulting figures are saved to disk, with the filenames
    saved to each parent's subtiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    if not layer.sub_layer:
        raise ValueError('no sub_layer set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_subgains{-user_tag}-{layer_level}-{parent_index}
    fnametmplt = '%s/plot_subgains%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    if verbose:
        print >> sys.stdout, "creating volume subtiles plots for level %i:" %(
            layer.level)
    for ii,parent in enumerate(layer.parents):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(layer.parents)),
            sys.stdout.flush()
        mfig = plot_subgains(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label,
            layer.sub_layer.x_param, layer.sub_layer.x_param.label,
            layer.sub_layer.y_param, layer.sub_layer.y_param.label,
            test_threshold, ref_threshold, min_ninj=min_ninj,
            test_livetime=test_livetime, ref_livetime=ref_livetime,
            test_label=test_label, ref_label=ref_label,
            colormap=colormap, maxgain=maxgain, mingain=mingain,
            logx=layer.plot_log_x, logy=layer.plot_log_y,
            sub_logx=layer.sub_layer.x_distr == 'log10',
            sub_logy=layer.sub_layer.y_distr == 'log10',
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)

        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        parent.subtiles_plot = mfig

    if verbose:
        print >> sys.stdout, ""
