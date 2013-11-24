from __future__ import print_function, division
import os


def savefigure(name, loc, fig, Ext=['pdf', 'eps', 'png']):
    '''
    Saves figure to location given by rootloc/<ext>/<name>.<ext>
    '''

    for ext in Ext:
        extloc = os.path.join(loc, ext)
        if not os.path.exists(extloc):
            os.makedirs(extloc)

    for ext in Ext:
        aname = name + '.' + ext
        saveloc = os.path.join(loc, ext, aname)
        fig.savefig(saveloc)
