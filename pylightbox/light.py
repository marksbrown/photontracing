'''
light.py

Author : Mark S. Brown
First Commit : 3rd November 2013

Description : In this module functionality required for ray tracing in geometry
defined in _box_ is contained.

'''
from __future__ import print_function, division
from numpy import array, dot, sin, cos, pi, ones, arccos, floor, ptp
from numpy import abs, random, linalg, zeros, where, sqrt, arcsin, ones, tan, isnan, inner
from numpy import shape, cumsum, arctan2, vstack
import os
from scipy import stats
from pandas import DataFrame, read_csv
from const import *

def lazydot(u, v):  # quicker than np.dot for the way we've set up our code
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])


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


def RotateVector(v, phi=0, theta=0, psi=0, verbose=0):
    '''
    rotate vector 'v' using Euler Angles
    '''

    R = ((
        (cos(theta) * cos(psi)),
        (-cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)),
        (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)),
    ), (
        (cos(theta) * sin(psi)),
        (cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi)),
        (-sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)),
    ), (
        (-sin(theta)),
        (sin(phi) * cos(theta)),
        (cos(phi) * cos(theta)),
    ))

    if verbose > 0:
        print(shape(R), shape(v), phi / Degrees, theta / Degrees)
        print("matrix to rotate is", v)
    return dot(R, array(v).T).T


def IsotropicSource(N, Pos=[0, 0, 0]):
    '''
    Returns a list of initial photons of size N
    '''
    return DataFrame([{"direction": d, "energy": 1 / N, "facet": -1,
                       "position": array(Pos), "time": 0, "angle": 0,
                       "photonstatus": "Trapped"} for d in RandomPointsOnASphere(N)])


def SampledDirection(
    N=1, loc=0 * Degrees, scale=1.3 * Degrees, dist=stats.norm,
        surfacenormal=[0, 0, 1], verbose=0):
    '''
    Generate N beams with profile given by the distribution with
    known scale and loc parameters

    Default gives normal distribution with a standard deviation of 1.3 degrees
    (corresponding to a polished surface - Moses2010)
    '''

    Theta = dist(loc=loc, scale=scale).rvs(N)  # sampled theta
    Phi = random.uniform(-1, 1, N) * pi  # uniform phi

    X = sin(Theta) * cos(Phi)
    Y = sin(Theta) * sin(Phi)
    Z = cos(Theta)

    newvectors = vstack((X, Y, Z)).T

    if verbose > 0:
        print("New Vectors with shape", shape(newvectors))

    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])

    if verbose > 0:
        print("shape of surface normals is", shape(surfacenormal))
    return (
        RotateVector(
            newvectors,
            phi(surfacenormal),
            theta(surfacenormal),
            verbose=verbose)
    )


def LobeReflection(
        N=1, stddev=1.3 * Degrees, surfacenormal=[0, 0, 1], verbose=0):
    return SampledDirection(N, loc=0, scale=stddev, dist=stats.norm,
                            surfacenormal=surfacenormal, verbose=verbose)


def LambertianReflection(N=1, surfacenormal=[0, 0, 1], verbose=0):
    return SampledDirection(N, loc=0, scale=0.5, dist=stats.cosine,
                            surfacenormal=surfacenormal, verbose=verbose)


def NewDirections(M, Pos):
    Indices = random.randint(0, len(Pos), M)
    return Pos[Indices]


def RandomPointsOnASphere(N, hemisphere=False):
    '''
    Generates random points on a sphere
    or on a hemisphere (default is sphere)
    '''
    Values = []

    while len(Values) < N:
        x1 = 2 * random.random() - 1
        x2 = 2 * random.random() - 1

        if x1 ** 2 + x2 ** 2 < 1:
            Values.append((x1, x2))

    x1, x2 = zip(*Values)
    x1 = array(x1)
    x2 = array(x2)
    x = 2 * x1 * sqrt(1 - x1 ** 2 - x2 ** 2)
    y = 2 * x2 * sqrt(1 - x1 ** 2 - x2 ** 2)
    if hemisphere:
        z = abs(1 - 2 * (x1 ** 2 + x2 ** 2))
    else:
        z = 1 - 2 * (x1 ** 2 + x2 ** 2)

    return vstack((x, y, z)).T


def NearestFacet(ph, aBox, Threshold=1e-15, verbose=0):
    '''
    returns the nearest facet normal for dataframe `ph'

    This function is setup in a vectorised fashion to ensure the thing
    is bloody quicker than the previous version
    '''
    nds = zeros(len(ph))
    Facets = zeros(len(ph))
    DistanceTo = ones(len(ph)) / Threshold

    # currently needed...
    Directions = array([list(adir) for adir in ph.direction])
    # ...for speeding up dot
    Positions = array([list(adir) for adir in ph.position])

    # iterates over each face
    for (i, sn), sp in zip(enumerate(aBox.normals), aBox.points):

        #ndots = ph.direction.apply(dot,args=[sn])
        #ndots = ph.direction.apply(lazydot,args=[sn])
        ndots = dot(Directions, sn)

        #dmin = ph.position.apply(dot,args=[sn])
        #dmin = ph.position.apply(lazydot,args=[sn])
        dmin = dot(Positions, sn)

        #dmin -= dot(sn,sp)
        dmin -= lazydot(sn, sp)

        #dmin = dmin.apply(abs)
        dmin = abs(dmin)

        DistanceToFacet = dmin / ndots
        if verbose > 1:
            print(i, list(DistanceToFacet))
        Conditions = (ndots > 0) & (DistanceToFacet < DistanceTo)
        Facets = where(Conditions, i, Facets)
        nds = where(Conditions, ndots, nds)
        DistanceTo = where(Conditions, DistanceToFacet, DistanceTo)

    ph["surfacenormal"] = [array(aBox.normals[int(j)]) for j in Facets]
    ph["distanceto"] = DistanceTo
    if verbose > 1:
        print(ph.distanceto)
    ph["facet"] = Facets
    ph["ndots"] = nds
    return ph

    #surfacenormal = [array(aBox.normals[int(j)]) for j in Facets]
    # return surfacenormal,DistanceTo,Facets,nds


def EscapeStatus(ph, aBox, verbose=0, **kwargs):
    '''
    Photons arriving at a surface will change status to 'trapped','escaped'
    or 'absorbed' based on order of events
    1st test : critical angle (trapped if angle is within)
    2nd test : fresnel reflection
    3rd test : reflectivity parameter - this WILL override everything else
    '''

    reflectivity = kwargs.get('reflectivity', True)
    fresnel = kwargs.get('fresnel', True)

    if verbose > 1:
        print("Fresnel is", fresnel)
        print("Reflectivity is", reflectivity)

    photonstatus = array(["Trapped"] * len(ph))  # default status

    CritAngles = ph.facet.apply(aBox.Crit)  # critical angle for each photon
    escapestatus = ph["angle"] < CritAngles

    if fresnel:
        Fresnel = aBox.Fresnel(ph)
        escapestatus = (
            array(Fresnel) < random.uniform(size=len(ph))) & escapestatus

    if reflectivity:
        # gets reflectivity of each photon
        Reflectivities = ph.facet.apply(aBox.Ref)
        escapestatus = (
            array(Reflectivities) < random.uniform(size=len(ph))) & escapestatus

    photonstatus[escapestatus] = "Escaped"

    return photonstatus


def NewDirection(ph, aBox, verbose=0):
    '''
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    '''

    param = ph.facet.apply(aBox.GetUnified)  # unified parameters for each list

    def firsttrue(param):
        unifiednames = ['specular', 'lobe', 'backscatter', 'lambertian']
        for j, p in enumerate(param):
            if p:
                return unifiednames[j]

    ph['whichreflection'] = [firsttrue(ru < up) for (ru, up)
                             in zip(random.uniform(size=len(ph)), param)]

    def newdirection(key, grp):
        if key == "specular":
            return (
                # updates to specular direction
                grp.direction - 2 * grp.ndots * grp.surfacenormal
            )
        elif key == 'lobe':
            return (
                LobeReflection(len(grp), surfacenormal=list(grp.surfacenormal))
            )
        elif key == 'backscatter':
            return -1 * grp.direction
        elif key == 'lambertian':
            return (
                LambertianReflection(len(grp), surfacenormal=grp.surfacenormal)
            )
        else:
            pass  # add more kinds of reflection!

    l = [newdirection(key, grp) for key, grp in ph.groupby('whichreflection')]
    return [item for sublist in l for item in sublist]


def MoveToNearestFacet(ph, aBox, sp, verbose=0):
    '''
    Boundary process function in order we :
    1)Move to the nearest face (updating the position and time)
    2)Determine the escape status
    3)Find the new direction from the UNIFIED parameters
    '''

    ph.angle = ph.ndots.apply(arccos)
    ph.position += ph.distanceto * ph.direction  # updates position to boundary
    ph.time += aBox.n * ph.distanceto / SpeedOfLight  # updates travel time

    ph["photonstatus"] = EscapeStatus(ph, aBox, **sp)
    ph.direction = NewDirection(ph, aBox, verbose=verbose)

    return ph


def ForeverReflective(ph, aBox, runs=1, verbose=0, **kwargs):
    '''
    Photons never escape! (mwhahaha)
    '''
    surfaceproperties = kwargs.get(
        'surface', {'reflectivity': True, 'fresnel': False})

    for runnum in range(runs):
        ph = NearestFacet(ph, aBox, verbose=verbose)  # Calculates nearest face
        # Moves to nearest face
        ph = MoveToNearestFacet(ph, aBox, sp=surfaceproperties)

    return ph


def Polished(ph, aBox, runs=1, verbose=0,
             fetchall=False, MaxRepeat=5, **kwargs):
    '''
    Box is perfectly polished (only specular reflection)
    MaxRepeat states if the length of the escaped dataframe
    doesn't increase, we return early
    '''
    surfaceproperties = kwargs.get(
        'surface', {'reflectivity': True, 'fresnel': False})

    for runnum in range(runs):
        if verbose > 0:
            print(runnum, "::", end=" ")
            for key, grp in ph.groupby("photonstatus"):
                print(key, ":", str(len(grp)).ljust(10), end=" ")
            print("")

        ph = NearestFacet(
            ph[ph.photonstatus == "Trapped"],
            aBox,
            verbose=verbose)
        # Moves to nearest face
        ph = MoveToNearestFacet(ph, aBox, sp=surfaceproperties)
        try:
            Escaped = Escaped.append(ph[ph.photonstatus == "Escaped"])
            if len(Escaped) == LengthOfEscaped:
                EscapeEarlyCounter += 1  # no additional photons are leaving!
            LengthOfEscaped = len(Escaped)

        except NameError:
            Escaped = ph[ph.photonstatus == "Escaped"]
            LengthOfEscaped = len(Escaped)
            EscapeEarlyCounter = 0

        if EscapeEarlyCounter == MaxRepeat:
            if verbose > 0:
                print("Escaping on run", runnum, "where maximum runs is", runs)
            break

    if fetchall:
        return ph.append(Escaped)
    else:
        return Escaped


def Unified(ph, aBox, runs=1, verbose=0, fetchall=False, MaxRepeat=5):
    '''
    Light within the unified model case (woo woo)
    '''
    for runnum in range(runs):
        if verbose > 1:
            print(ph.groupby("photonstatus").size())

        # ph[ph.photonstatus == "Trapped"].direction =
        #    ph[ph.photonstatus == "Trapped"].direction.apply(

        ph = NearestFacet(ph[ph.photonstatus == "Trapped"], aBox, verbose=0)
        ph = MoveToNearestFacet(ph, aBox)  # Moves to nearest face
        try:
            Escaped = Escaped.append(ph[ph.photonstatus == "Escaped"])
            if len(Escaped) == LengthOfEscaped:
                EscapeEarlyCounter += 1  # no additional photons are leaving!
            LengthOfEscaped = len(Escaped)

        except NameError:
            Escaped = ph[ph.photonstatus == "Escaped"]
            LengthOfEscaped = len(Escaped)
            EscapeEarlyCounter = 0

        if EscapeEarlyCounter == MaxRepeat:
            if verbose > 0:
                print("Escaping on run", runnum, "where maximum runs is", runs)
            break

    if fetchall:
        return ph.append(Escaped)
    else:
        return Escaped

#


def PlotAngleAndTime(df, axis1, axis2, dt=1, timerange=(
        0, 500), da=1, anglerange=(0, 90), **kwargs):
    '''
    Plots the angle and time histograms on axis1 and axis2 respectively
    '''

    anglebins = floor(ptp(anglerange) / da)
    timebins = floor(ptp(timerange) / dt)

    defaulthist = {'histtype': 'stepfilled', 'alpha': 1.0}
    # override default args
    histkwargs = dict(defaulthist.items() + kwargs.items())

    for s, grp in df.groupby("photonstatus"):
        lbl = histkwargs.pop('label', s)  # gets label or defaults to s
        grp.angle /= Degrees  # turns into degrees
        axis1.hist(grp.angle, bins=anglebins, range=anglerange,
                   weights=grp.energy, label=lbl, **histkwargs)
        grp.time /= ps  # turns into ps

        axis2.hist(grp.time, bins=timebins, range=timerange,
                   weights=grp.energy, label=lbl, **histkwargs)

    axis1.grid(True)
    axis2.grid(True)
    axis1.set_xlim(0, 90)
    axis1.set_xlabel("Angle (Degrees)")
    axis2.set_xlabel("Time (ps)")
    axis1.set_ylabel("Energy Density")
    axis2.set_ylabel("Energy Density")
    leg = axis1.legend()
    leg.set_title("Configuration")
    leg = axis2.legend()
    leg.set_title("Configuration")
    return axis1, axis2
