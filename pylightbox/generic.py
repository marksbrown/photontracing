from __future__ import print_function, division
import os


def savefigure(name, loc, fig, Ext=['pdf', 'eps', 'png']):
    '''
    Saves figure to location given by rootloc/<ext>/<name>.<ext>
    '''
    if not os.path.exists(loc):
        os.mkdir(loc)

    for ext in Ext:
        extloc = os.path.join(loc, ext)
        if not os.path.exists(extloc):
            os.mkdir(extloc)

    for ext in Ext:
        aname = name + '.' + ext
        saveloc = os.path.join(loc, ext, aname)
        fig.savefig(saveloc)


def tocube(ax, anum=1):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)


def SaveData(df, name, key, loc):
    '''
    Saves data to chosen loc
    '''
    if not os.path.exists(loc):
        os.mkdir(loc)

    fn = "".join([name, '_', str(key), ".", "csv"])
    saveto = os.path.join(loc, fn)
    df.to_csv(saveto)


def LoadData(key, loc):
    '''
    Loads list of data files from chosen loc
    '''

    return {aloc.split('.')[0].split('_')[-1]: read_csv(os.path.join(loc, aloc), index_col=0)
            for aloc in os.listdir(loc) if aloc[-3:] == 'csv' and aloc.find(key) >= 0}
