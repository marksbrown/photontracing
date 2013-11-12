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
from numpy import shape, cumsum, arctan2, dstack, newaxis, tile, hstack, mean
import os
from scipy import stats
from pandas import DataFrame, read_csv
from .const import *


# tmp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import subplots, show
from . import box as Box


def lazydot(u, v):
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])


def _RotateVector(v, phi=0, theta=0, psi=0, verbose=0):
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
    return dot(v, R)


def RotateVectors(vectors, surfacenormal=[0, 0, 1], verbose=0):
    '''
    Rotate vectors
    '''
    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])

    return _RotateVector(vectors,
                         phi(surfacenormal),
                         theta(surfacenormal),
                         verbose=verbose)


def _SampledDirection(N, loc, scale, dist, surfacenormal, verbose=0):
    '''
    Generate N beams with profile given by the distribution with
    known scale and loc parameters
    '''

    Theta = dist(loc=loc, scale=scale).rvs(N)  # sampled theta
    Phi = random.uniform(-1, 1, N) * pi  # uniform phi

    X = sin(Theta) * cos(Phi)
    Y = sin(Theta) * sin(Phi)
    Z = cos(Theta)

    newvectors = dstack((X, Y, Z))[0, ...]

    return RotateVectors(newvectors, surfacenormal, verbose)


def _RandomPointsOnASphere(N, hemisphere=False, split=False):
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

    return dstack((x, y, z))[0, ...]


def SpecularReflection(olddirection, surfacenormal, ndots, verbose=0):
    '''
    Specular (mirror-like) Reflection
    '''    
    newdirection = olddirection - 2 * ndots[:, newaxis] * surfacenormal

    if verbose > 1:
        print("--Update Direction--")
        for od, nd in zip(olddirection, newdirection):
            print("Old direction :", od)
            print("New direction : ", nd)

    return newdirection


def LobeReflection(N=1, surfacenormal=[0, 0, 1], stddev=1.3 * Degrees, verbose=0):
    '''
    Gives normal distribution with a standard deviation of 1.3 degrees
    (corresponding to a polished surface - Moses2010)
    '''
    return _SampledDirection(N, loc=0, scale=stddev, dist=stats.norm,
                             surfacenormal=surfacenormal, verbose=verbose)


def LambertianReflection(N=1, surfacenormal=[0, 0, 1], verbose=0):
    '''
    Gives Lambertian distribution
    '''
    return _SampledDirection(N, loc=0, scale=0.5, dist=stats.cosine,
                             surfacenormal=surfacenormal, verbose=verbose)


def IsotropicReflection(N=1, surfacenormal=[0, 0, 1], verbose=0):
    '''
    no preferred direction hemispherical emission
    '''
    newvectors = _RandomPointsOnASphere(N, hemisphere=True)

    if verbose > 0:
        print("shape is", shape(newvectors))
    if verbose > 1:
        for avec in newvectors:
            print(avec)

    return RotateVectors(newvectors, surfacenormal, verbose)


def IsotropicSource(N, Pos=[0, 0, 0]):
    '''
    Returns a list of initial photons of size N
    direction, position, times
    '''
    return _RandomPointsOnASphere(N), array(list([Pos]) * N), zeros(N)


def NearestFace(Directions, Positions, aBox, verbose=0, threshold=1e-15):
    '''
    returns the distance to the nearest face, the nearest face index and
    the angle with respect to the surface normal
    '''

    nds = zeros(len(Directions))
    Faces = zeros(len(Directions))
    DistanceTo = ones(len(Directions)) / threshold

    # iterates over each face
    for (i, sn), sp in zip(enumerate(aBox.normals), aBox.points):
        ndots = dot(Directions, sn)
        dmin = dot(Positions, sn)
        dmin -= lazydot(sn, sp)
        dmin = abs(dmin)

        DistanceToFace = dmin / ndots
        Conditions = (ndots > 0) & (DistanceToFace < DistanceTo)
        Faces = where(Conditions, i, Faces)
        nds = where(Conditions, ndots, nds)
        DistanceTo = where(Conditions, DistanceToFace, DistanceTo)

    if verbose > 1:
        print("--Nearest Face--")
        print("face index : distanceto")
        for f, dst in zip(Faces, DistanceTo):
            print(f, ":", dst / mm)

    return Faces, DistanceTo, nds


def EscapeStatus(faces, ndots, aBox,
                 reflectivity=True, fresnel=True, verbose=0):
    '''
    Photons arriving at a surface will change status to 'trapped','escaped'
    or 'absorbed' based on order of events
    1st test : critical angle (trapped if angle is within)
    2nd test : fresnel reflection
    3rd test : reflectivity parameter - this WILL override everything else
    '''

    CritAngles = zeros(shape(faces))
    for uniqueface in set(faces):
        CritAngles[faces == uniqueface] = aBox.Crit(uniqueface)

    angles = array([arccos(aval) for aval in ndots])
    escapestatus = angles < CritAngles

    if verbose > 1:
        print("--Critical--")
        print("incident angle : Escaped?")
        for ang, esc in zip(angles, escapestatus):
            print(ang / Degrees, ":", esc)
        print("Escaping", sum(escapestatus == True))

    if fresnel:
        Fresnel = aBox.Fresnel(faces, angles)
        ru = random.uniform(size=len(faces))
        escapestatus = (array(Fresnel) < ru) & escapestatus

        if verbose > 1:
            print("--Fresnel--")
            print("Reflectance : Escaped?")
            for fr, esc in zip(Fresnel, escapestatus):
                print(fr, ":", esc)
            print("Escaping", sum(escapestatus == True))

    if reflectivity:
        Reflectivities = zeros(shape(faces))
        for uniqueface in set(faces):
            Reflectivities[faces == uniqueface] = aBox.Ref(uniqueface)

        ru = random.uniform(size=len(faces))
        escapestatus = (array(Reflectivities) < ru) & escapestatus

        if verbose > 1:
            print("--Reflectivity--")
            print("Reflectivity : Escaped?")
            for rf, esc in zip(Reflectivities, escapestatus):
                print(rf, ":", esc)
            print("Escaping", sum(escapestatus == True))

    return escapestatus  # if true, photon escapes


def UpdateDirection(olddirection, faces, ndots, aBox, verbose=0):
    '''
    Updates direction (and time!) of photons
    '''

    newdirection = zeros(shape(olddirection))
    
    for uniqueface in set(faces):
        if not any(faces == uniqueface):
            continue
        else:
            pass
            #print(uniqueface," has ", sum(faces == uniqueface), " photons incident")
        surfacenormal = aBox.normals[uniqueface]
        od = olddirection[faces == uniqueface,...]
        nds = ndots[faces == uniqueface]
        newdirection[faces == uniqueface,...] = SpecularReflection(od, surfacenormal, nds, verbose=verbose)

    return newdirection

#def UpdateDirection(olddirection, faces, ndots, aBox, verbose=0):
#    '''
#    Updates direction (and time!) of photons
#    '''
#    
#    surfacenormals = array([aBox.normals[int(j)] for j in faces])
#           
#    return SpecularReflection(olddirection, surfacenormals, ndots, verbose=verbose)    



def UpdateUnifiedDirection(olddirection, faces, ndots, aBox, verbose=0):
    '''
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    '''
    
    surfacenormals = array([aBox.normals[int(j)] for j in faces])

    unifiedparameters = zeros(shape(faces))
    for uniqueface in set(faces):
        unifiedparameters[faces == uniqueface] = aBox.GetUnified(uniqueface)
        
    def firsttrue(param):
        unifiednames = ['specular','lobe','backscatter','lambertian']
        for j,p in enumerate(param):
            if p:
                return unifiednames[j]

    whichreflection = [firsttrue(ru < up) for (ru,up) 
                in zip(random.uniform(size=len(faces)), unifiedparameters)]
    
    def getnewdirection(key, olddirection, ndots, surfacenormal):
        if key == "specular":
            return SpecularReflection(olddirection, surfacenormal, ndots, verbose=verbose)    
        elif key == 'lobe':
            return LobeReflection(len(ndots), surfacenormal)
        elif key == 'backscatter':
            return -1*olddirection
        elif key == 'lambertian':
            return LambertianReflection(len(ndots),surfacenormal)
        else:
            print("Unknown Reflection type!")
            return 
    
    #TODO get this all bloody working
    #Here's how I see it working : we pass by surface normal grouped such that we
    #Can general large N of new directions rotated easily. This does have a cost
    #when it comes to specular but we can worry about that later. For now, this needs
    #To be set up correctly such that we only pass the information needs to the above
    #function and everything else is kept out of the way. We also need to (SOMEHOW!)
    #collect the _newdirections_ in a sane manner please - right now it will not work!

    ##good luck :)
    newdirections = tile(shape(faces)) 
    for uniqueface in set(faces):
        for aref in set(whichreflection): #faces x numunified (6 x 4 groups for a cube!)
            newdirections[faces == uniqueface] = getnewdirection(aref, 
                         olddirection[faces == uniqueface],
                         ndots[faces == uniqueface],
                         array(aBox.normals[uniqueface]))
        
        
    l = [newdirection(key,grp) for key,grp in ph.groupby('whichreflection')]
    return [item for sublist in l for item in sublist]

def UpdatePosition(oldposition, distanceto, oldtime, directions, aBox, verbose=0):
                   
    '''
    Moves photons to new position
    '''

    newposition = oldposition + distanceto[:, newaxis] * directions
    newtime = oldtime + aBox.n * distanceto / SpeedOfLight

    if verbose > 1:
        print("--Update Position--")
        for np, nt in zip(newposition, newtime):
            print("New position : ", array(np) / mm)
            print("Updated time : ", nt / ps, "ps")

    return newposition, newtime


def ToDataFrame(directions, positions, times, faces):

    adict = [{"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
             "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
              "face": fc, "time": atime}
             for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime in
             zip(positions, directions, faces, times)]

    return DataFrame(adict)


def IsotropicSource(N, Pos=[0, 0, 0]):
    '''
    Returns a list of initial photons of size N
    direction, position, times
    '''
    return _RandomPointsOnASphere(N), array(list([Pos]) * N), zeros(N)


def LightinaBox(idir, ipos, itime, aBox, runs=1, verbose=0, **kwargs):
    '''
    Meh
    '''

    reflectivity = kwargs.get('reflectivity', True)
    fresnel = kwargs.get('fresnel', True)

    ProcessedPhotons = []

    for runnum in range(runs):
        if runnum == 0:
            directions = idir
            positions = ipos
            times = itime

        faces, distanceto, ndots = NearestFace(directions, positions,
                                               aBox, verbose=verbose)

        positions, times = UpdatePosition(positions, distanceto, times,
                                          directions, aBox, verbose=verbose)

        directions = UpdateDirection(directions, faces, ndots,
                                     aBox, verbose=verbose)

        est = EscapeStatus(faces, ndots, aBox, fresnel=fresnel,
                           reflectivity=reflectivity, verbose=verbose)

        if verbose > 0:
            print("--Photons Escaped--")
            print(runnum, ": Escaped", sum(est))

        ProcessedPhotons += [
            {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
             "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
             "face": fc, "time": atime, "ndots": nds, "photonstatus": "Escaped"}
            for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
            zip(positions[est], directions[est], faces[est], times[est], ndots[est])]
                                  

        directions = directions[est == False]
        positions = positions[est == False]
        times = times[est == False]

    # adds on all remaining 'trapped' photons
    ProcessedPhotons += [
        {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
         "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
         "face": fc, "time": atime, "ndots": nds, "photonstatus": "Trapped"}
        for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
        zip(positions, directions, faces, times, ndots)]
                              

    return ProcessedPhotons
