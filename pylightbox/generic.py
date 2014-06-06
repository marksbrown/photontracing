from __future__ import print_function, division
import os
from .const import *
from numpy import ptp, array


def savefigure(name, loc, fig, Ext=('pdf', 'eps', 'png')):
    """
    Saves figure to location given by rootloc/<ext>/<name>.<ext>
    """

    for ext in Ext:
        extloc = os.path.join(loc, ext)
        if not os.path.exists(extloc):
            os.makedirs(extloc)
        
        aname = name + '.' + ext
        saveloc = os.path.join(extloc, aname)
        fig.savefig(saveloc)


def tocube(axis, anum=1):
    axis.set_xlim(-anum, anum)
    axis.set_ylim(-anum, anum)
    axis.set_zlim(-anum, anum)


def labelaxes(axis, defaultunit=mm):
    axis.set_xticklabels(axis.get_xticks() / defaultunit)
    axis.set_yticklabels(axis.get_yticks() / defaultunit)
    axis.set_zticklabels(axis.get_zticks() / defaultunit)

    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")


def PlotTime(axis, df, verbose=0, **kwargs):

    timerange = kwargs.pop('timerange',(0, 1000))
    xlabel = kwargs.pop("xlabel","time (ns)")
    ylabel = kwargs.pop("ylabel","Energy Density")
    dt = kwargs.pop('dt',1)
    Bins = ptp(timerange)/dt

    if verbose > 0:  
        print("Energy in plot is :",sum(df.energy)*1e2,"%")
    
    axis.hist(array(df.time), range=timerange,bins=Bins, 
                weights=array(df.energy),**kwargs)
    
    axis.grid(True)
    axis.set_xlim(timerange)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    

def PlotAngle(axis, df, verbose=0, **kwargs):
    
    axis.hist(df.angle/Degrees,range=(0,90),bins=91, weights=df.energy,**kwargs)
    axis.grid(True)
    axis.set_xlabel("Angle (Degrees)")
    axis.set_ylabel("frequency")

    
