import matplotlib
import matplotlib.pyplot as mp
import numpy as np


def decision_boundary_2d(classifier, labels_values, resx, resy):
    mp.scatter(*np.array([x[1] for x in labels_values if x[0] == 1]).T,
               s=2, linewidth=0.)
    mp.scatter(*np.array([x[1] for x in labels_values if x[0] != 1]).T, c='r',
               s=2, linewidth=0.)
    minmax = lambda x: (np.min(x), np.max(x))
    minx, maxx = minmax([x[1][0] for x in labels_values])
    miny, maxy = minmax([x[1][1] for x in labels_values])
    xcoords = np.arange(minx, maxx, resx)
    ycoords = np.arange(miny, maxy, resy)
    xval = np.zeros((len(ycoords), len(xcoords)))
    yval = np.zeros((len(ycoords), len(xcoords)))
    zval = np.zeros((len(ycoords), len(xcoords)))
    for yind, ycoord in enumerate(ycoords):
        for xind, xcoord in enumerate(xcoords):
            val = classifier.predict(np.array([xcoord, ycoord]))
            xval[yind, xind] = xcoord
            yval[yind, xind] = ycoord
            zval[yind, xind] = val[0][0] * val[0][1]
    cs = mp.contour(xval, yval, zval)
    mp.clabel(cs, fontsize=9, inline=1, c='k')
    mp.show()
