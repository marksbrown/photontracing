from __future__ import print_function, division
import os
from .const import *


def savefigure(name, loc, fig, Ext=['pdf', 'eps', 'png']):
    '''
    Saves figure to location given by rootloc/<ext>/<name>.<ext>
    '''

    for ext in Ext:
        extloc = os.path.join(loc, ext)
        if not os.path.exists(extloc):
            os.makedirs(extloc)
        
        aname = name + '.' + ext
        saveloc = os.path.join(extloc, aname)
        fig.savefig(saveloc)

def tocube(axis, anum=1):
    axis.set_xlim(-anum,anum)
    axis.set_ylim(-anum,anum)
    axis.set_zlim(-anum,anum)


def labelaxes(axis, defaultunit=mm):
    axis.set_xticklabels(axis.get_xticks() / defaultunit)
    axis.set_yticklabels(axis.get_yticks() / defaultunit)
    axis.set_zticklabels(axis.get_zticks() / defaultunit)

    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
