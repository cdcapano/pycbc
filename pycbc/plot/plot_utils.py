#! /usr/bin/env python

import os
import functools
import numpy
import matplotlib
from matplotlib import pyplot

def drop_trailing_zeros(num):
    """
    Drops the trailing zeros in a float that is printed.
    """
    txt = '%f' %(num)
    txt = txt.rstrip('0')
    if txt.endswith('.'):
        txt = txt[:-1]
    return txt

def get_signum(val, err, max_sig=numpy.inf):
    """
    Given an error, returns a string for val formated to the appropriate
    number of significant figures.
    """
    coeff, pwr = ('%e' % err).split('e')
    if pwr.startswith('-'):
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr -= 1
        pwr = min(pwr, max_sig)
        tmplt = '%.' + str(pwr) + 'f'
        return tmplt % val
    else:
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr += 1
        # if the error is large, we can sometimes get 0;
        # adjust the round until we don't get 0 (assuming the actual
        # value isn't 0)
        return_val = round(val, -pwr)
        if val != 0.:
            loop_count = 0
            max_recursion = 100
            while return_val == 0.:
                pwr -= 1
                return_val = round(val, -pwr)
                loop_count += 1
                if loop_count > max_recursion:
                    raise ValueError("Maximum recursion depth hit! Input " +\
                        "values are: val = %f, err = %f" %(val, err))
        return drop_trailing_zeros(return_val)

def get_color_grayscale(clr):
    """
    Converts the given color to grayscale. The equation used to do the
    conversion comes from:
    en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

    Parameters
    ----------
    clr: tuple
        A color tuple. Should be in the form of (R, G, B, A).

    Returns
    -------
    grayscale: float
        A number between 0. and 1. representing the grayscale, with 0 = black,
        1 = white.
    """
    r, g, b, a = clr
    return 0.299*r + 0.587*g + 0.114*b

def ColorBarLog10Formatter(y, pos):
    """
    Formats y=log10(x) values for a colorbar so that they appear as 10^y.
    """
    return "$10^{%.2f}$" % y

def create_bounded_colorbar_formatter(minlim, maxlim, formatter=None):
    """
    Creates a frozen wrapper around the given colorbar formatter function
    that adds a < (>) symbol to the ticklabel if the y value is <= (>=) minlim
    (maxlim).
    """
    if formatter is None:
        # FIXME: I don't know how colorbar generates it's default formatter,
        # so I'm just creating a dummy colorbar to get it
        dumbfig = pyplot.figure()
        dumbax = dumbfig.add_subplot(111)
        dumbsc = dumbax.scatter(range(2), range(2), c=(0.9*minlim, 1.1*maxlim),
            vmin=minlim, vmax=maxlim)
        dumbcb = dumbfig.colorbar(dumbsc)
        formatter = dumbcb.formatter
        del dumbfig, dumbax, dumbsc, dumbcb
    def bounded_colorbar_formatter(y, pos, formatter, minlim, maxlim):
        tickstr = formatter(y, pos)
        if tickstr.startswith('$'):
            tickstr = tickstr[1:]
            add_dollar = '$'
        else:
            add_dollar = ''
        #if y < minlim:
        #    tickstr = '< %s' %(tickstr)
        #if y > maxlim:
        #    tickstr = '> %s' %(tickstr)
        return add_dollar + tickstr
    return pyplot.FuncFormatter(functools.partial(
        bounded_colorbar_formatter, formatter=formatter, minlim=minlim,
        maxlim=maxlim))

def empty_plot(ax, message="Nothing to plot"):
    """
    Creates an empty plot on the given axis.
    """
    # ensure the axis background is white, so the text can be read
    ax.set_axis_bgcolor('w')
    return ax.annotate(message, (0.5, 0.5))

def empty_hatched_plot(ax, hatch="x"):
    """
    Creates an empty plot on the given axis consisting entirely of a hatched
    background.
    """
    tile = ax.fill((0,1,1,0), (0,0,1,1), hatch=hatch, facecolor='w')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    return tile


#
#
#   Utilities for creating html image maps
#
#
class ClickableElement:
    """
    Class to store additional information about an element in a matplotlib
    plot that is needed to create an image map area of it in an html page.

    Parameters
    ----------
    element: matplotlib element
        The element (e.g., a polygon) in a matplotlib axis overwhich we will
        create an image map area.
    shape: str
        The shape of the image map area. Valid shapes are given by the
        validshapes attribute.
    link: str
        The location of the file that the clickable area will link to.
    tag: str
        Title to appear over the clickable area during mouseover. This is
        added to the title and alt attributes of the area.
    data: Python object
        The object containing the data that the plot element was created from.
        For instance, if the plot element came from some data stored in an
        LSC Table row, data will be that row. This is useful if one needs
        to get additional information about the element, such as when
        building the html page that the clickable area points to.
    pixelcoords: numpy array
        The coordinates of the area, in display units. This should be a 2D
        array of x,y pairs. The x-values are measured from the left edge
        of the figure, the y-values from the top edge of the figure. (Note that
        this is different from matplotlib, in which the y-values are measured
        from the bottom edge.) To ensure proper formatting, pixelcoords can
        can only be set using self.set_pixelcoords().
    radius: int
        The radius of the clickable area, in display units. This should only
        be set if self's shape is circle.
    """
    element = None
    _shape = None
    _radius = None
    _pixelcoords = None
    data = None
    link = None
    tag = None
    validshapes = ['circle', 'rect', 'poly']
    
    def __init__(self, element, shape, data=None, link=None, tag=None,
        pixelcoords=None, radius=None):
        self.element = element
        self.set_shape(shape)
        if radius is not None:
            self.set_radius(radius)
        self.data = data
        self.link = link
        self.tag = tag
        if self.pixelcoords is not None:
            self.set_pixelcoords(pixelcoords)
        else:
            self._pixelcoords = None

    @property
    def shape(self):
        return self._shape

    def set_shape(self, shape):
        if not shape in self.validshapes:
            raise ValueError("unrecognized shape %s; options are %s" %(
                shape, ', '.join(self.validshapes)))
        self._shape = shape

    @property
    def radius(self):
        return self._radius

    def set_radius(self, radius):
        if self._shape != 'circle':
            raise ValueError("a radius should only be set if self's shape " +\
                "is a circle")
        self._radius = radius

    @property
    def pixelcoords(self):
        return self._pixelcoords

    def set_pixelcoords(self, coords):
        """
        Sets the pixelcoords attribute. The given coords must be a 2D array
        of x,y pairs corresponding to the pixel coordinates of the element.
        If self's shape is a rect, coords should be a 2x2 array in which the
        first row is the bottom left corner and the second row is the top
        right corner of the rectangle. If self's shape is circle, coords
        should be a 2x1 array giving the x,y location of the point. If self's
        shape is poly, coords should bw a 2xN array giving the vertices of
        the polygon, where N > 2.
        """
        # check that we have sane options
        if len(coords.shape) != 2:
            raise ValueError('coords should be a 2D array of x,y pairs')
        if self._shape == 'rect' and coords.shape != (2,2):
            raise ValueError('rect shape should have two x and two y ' +
                'coordinates, corresponding to the location of the vertices.')
        if self._shape == 'circle' and coords.shape != (1,2):
            raise ValueError('circle shape should only have a ' +
                'single x,y coordinate')
        if self._shape == 'poly' and coords.shape[0] < 3:
            raise ValueError('poly shape should have at least three x,y ' +
                'coordinates')
        # passed, set the coordinates
        self._pixelcoords = coords


class MappableFigure(matplotlib.figure.Figure):
    """
    A derived class of pyplot.figure.Figure, this adds additional
    attributes to Figure to make it easier to create an image map for it. The
    class has a get_pixelcoordinates function which will automatically get the
    pixel coordinates of all desired clickable elements formatted in the
    correct way for an image map. The class also has a savefig function that
    replace's the base class's savefig. This ensures that the proper
    coordinate transformations are carried out to get the pixel coordinates
    when the figure is renderred.

    Additional Parameters
    ---------------------
    figure_filename: str
        The name of the file the figure is saved to. This is set when savefig
        is called.
    saved_img_height: float
        The height of the figure's image in pixels when it is saved, set
        when savefig is called. This is needed for getting the proper
        coordinate locations when creating an image map. 
    saved_img_width: float
        Same as saved_img_height, but for width.
    clickable_elements: list
        A list of ClickableElement instances for which clickable areas will be
        created in an image map. All of the plot elements drawn on an axes
        that is attached to the figure.  To ensure this, a list may only be
        added via the set_clickable_elements function. To add more elements,
        use the add_clickable function.
    """
    _figure_filename = None
    _saved_img_height = None
    _saved_img_width = None
    _clickable_elements = []

    def __init__(self, *args, **kwargs):
        """
        Initializes _figure_filename, _saved_img_height, and _saved_img_width
        as None and _clickable_elements as an empty list. It then calls the
        base class's __init__ with the args and kwargs.
        """
        self._figure_filename = None
        self._saved_img_height = None
        self._saved_img_width = None
        self._clickable_elements = []
        # now initialize the figure
        super(MappableFigure, self).__init__(*args, **kwargs)

    def check_clickable_element(self, clickable):
        """
        Checks that the given clickable's element is in self's figure.
        If it is not, a ValueError is raised.
        """
        if clickable.element.figure != self:
            raise ValueError("clickable is not an element in self's figure")

    def add_clickable(self, clickable):
        self.check_clickable_element(clickable)
        self._clickable_elements.append(clickable)

    def set_clickable_elements(self, clickable_elements):
        map(self.check_clickable_element, clickable_elements)
        self._clickable_elements = clickable_elements

    @property
    def clickable_elements(self):
        return self._clickable_elements

    @property
    def saved_img_height(self):
        return self._saved_img_height

    @property
    def saved_img_width(self):
        return self._saved_img_width

    def get_pixel_coordinates(self, event):
        """
        Function to get the pixel coordinates of the clickable elements
        when the figure is saved.
        """
        self._saved_img_height = self.bbox.height
        self._saved_img_width = self.bbox.width
        for clickable in self._clickable_elements:
            ax = clickable.element.axes
            if ax not in self.axes:
                raise ValueError("clickable is not on an axes in this figure")
            pixel_coordinates = ax.transData.transform(clickable.element.xy)
            # pixel cooredinates are a 2D array; the first column is the
            # x coordinates, the second, the y coordinates
            # In html, the y-coordinate of an area is measured from the top
            # of the image, whereas matplotlib measures y from the bottom
            # of this image. For this reason, we have to adjust the y
            pixel_coordinates[:,1] *= -1.
            pixel_coordinates[:,1] += self.bbox.height
            clickable.set_pixelcoords(pixel_coordinates)

    def savefig(self, filename, **kwargs):
        """
        A wrapper around pyplot.savefig that will get the area coordinates
        of the clickable elements while saving. This needs to be done
        when the figure is rendered for saving to ensure the proper
        coordinates are retrieved. For more information, see:
        http://stackoverflow.com/a/4672015

        Also: the bbox_inches='tight' argument is not allowed if there are
        clickable elements, as this does not properly update the transformation
        matrices when the figure is trimmed (as of matplotlib version 1.4.1).
        """
        if 'bbox_inches' in kwargs and kwargs['bbox_inches'] == 'tight' and \
                self._clickable_elements != []:
            raise ValueError("bbox_inches=tight is not allowed when saving " +\
                "a mappable figure with clickable elements, as it will " +\
                "silently alter the pixel coordinates in the saved image. " +\
                "If you would like to save without the clickable elements, " +\
                "remove the clickable_elements on self [run " +\
                "self.set_clickable_elements([])], or use pyplot.savefig.")
        cid = self.canvas.mpl_connect('draw_event', self.get_pixel_coordinates)
        self._figure_filename = filename
        super(MappableFigure, self).savefig(filename, **kwargs)
        self.canvas.mpl_disconnect(cid)

    @property
    def saved_filename(self):
        """
        The filename of the saved figure. Only set when self.savefig is called.
        """
        return self._figure_filename

    def create_image_map(self, html_filename, view_width=1000,
            view_height=None):
        """
        Creates an image map of self. The pixelcoords and links of every
        element in self.clickable_elements must be populated, and
        self.saved_filename must not be None, meaning that the figure
        was saved to an image file.

        Parameters
        ----------
        html_filename: str
            The file name of the html page that the image map will be placed
            in. This is needed so that the proper relative path to the figure's
            file name can be constructed.
        view_width: {750, int}
            The width, in pixels, of the displayed image. To use the original
            size of the image, pass None.
        view_height: {None, int}
            The height, in pixels, of the displayed image. Default is None,
            in which case the height will automatically be set by the
            browser based on the view_width, such that the aspect ratio is
            preserved.

        Returns
        -------
        html: str
            HTML code describing the image link. This can be added to an
            html file to show the image map.
        """
        return _plot2imgmap(self, html_filename, view_width, view_height)
            

def figure(**kwargs):
    """
    Wrapper around pyplot's figure function that substitutes a
    matplotlib.figure.Figure with a MappableFigure in FigureClass.
    """
    if 'FigureClass' in kwargs:
        raise ValueError("this function only supports " +\
            "FigureClass=MappableFigure; use pyplot.figure() if you wish " +\
            "to use a different FigureClass")
    kwargs['FigureClass'] = MappableFigure
    return pyplot.figure(**kwargs)


def _plot2imgmap(mfig, html_filename, view_width=1000,
        view_height=None):
    """
    Creates the necessary html code to display a figure with an image map
    on an html page.

    Parameters
    ----------
    mfig: MappableFigure instance
        A instance of a MappableFigure. The clickable_elements list must
        be populated, and the figure must have been saved to a file via
        the mfig's savefig function.
    html_filename: str
        The file name of the html page that the image map will be placed
        in. This is needed so that the proper relative path to the figure's
        file name and the links can be constructed.
    view_width: {1000, int}
        The width, in pixels, of the displayed image. To use the original
        size of the image, pass None.
    view_height: {None, int}
        The height, in pixels, of the displayed image. Default is None,
        in which case the height will automatically be set by the
        browser based on the view_width, such that the aspect ratio is
        preserved.
    
    Returns
    -------
    html: str
        HTML code describing the image link. This can be added to an
        html file to show the image map.
    """
    if mfig.saved_filename is None:
        raise ValueError("mfig's saved_filename is None! Run " +
            "mfig's savefig to save the figure to disk.")
    figname = mfig.saved_filename
    # we need the fig filename relative to the html page's path
    figname = os.path.relpath(os.path.abspath(figname),
        os.path.dirname(os.path.abspath(html_filename)))
    img_height = int(mfig.saved_img_height)
    img_width = int(mfig.saved_img_width)
    # get the scale factors we need for the given view width and height
    scale_x = 1.
    scale_y = 1.
    if view_width is not None:
        scale_x = float(view_width)/img_width
        width_str = 'width="%i"' %(view_width)
        if view_height is None:
            scale_y = scale_x
    else:
        width_str = ''
    if view_height is not None:
        scale_y = float(view_height)/img_height
        if view_width is None:
            scale_x = scale_y
        height_str = 'height="%i"' %(view_height)
    else:
        height_str = ''
    scale = numpy.array([scale_x, scale_y])

    # construct the areas
    areas = []
    area_tmplt = '<area shape="%s" coords="%s" href="./%s" %s />'
    for clickable in mfig.clickable_elements:
        if clickable.shape == 'rect' or clickable.shape == 'poly':
            # for rectangle or polygon, the coordinates are just
            # x1,y1,x2,y2,...,xn,yn, where n=4 for rect
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
        elif clickable.shape == 'circle':
            if clickable.radius is None:
                raise ValueError("shape is circle, but no radius set!")
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
            coords += ',%i' %(clickable.radius)
        else:
            raise ValueError("unrecognized shape %s" %(clickable.shape))
        # format the link so that it is a relative path w.r.t. the html_page
        if clickable.link is None:
            raise ValueError("no link set!")
        link = os.path.relpath(os.path.abspath(clickable.link),
            os.path.dirname(os.path.abspath(html_filename)))
        # get any tags
        if clickable.tag is not None:
            tag_str = 'alt="%s" title="%s"' %(clickable.tag, clickable.tag)
        else:
            tag_str = ''
        areas.append(area_tmplt %(clickable.shape, coords, link, tag_str))

    # construct the map name from the figure's file name
    mapname = os.path.basename(figname)[:-4]

    # the map template
    tmplt = """
<div>
<img src="%s" class="mapper" usemap="#%s" border="0" %s %s />
</div>
<map name="%s">
%s
</map>
"""
    return tmplt %(figname, mapname, width_str, height_str, mapname,
        '\n'.join(areas))
