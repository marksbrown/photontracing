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
from numpy import shape, cumsum, arctan2, dstack, newaxis, tile, hstack, mean, issubdtype
from numpy import vstack, round, allclose
import os
from scipy import stats
from pandas import DataFrame, read_csv
from .const import *
import math


# tmp
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.pyplot import subplots, show
#from . import box as Box


def lazydot(u, v):
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])


def _RotateVector(v, phi=0, theta=0, psi=0, verbose=0):
    '''
    rotate vector 'v' using Euler Angles
    '''

    #http://stackoverflow.com/questions/19470955/warping-an-image-using-roll-pitch-and-yaw    
    warp_mat = lambda theta, psy, phi : array([[cos(theta)*cos(psy), cos(phi)*sin(psy)+sin(phi)*sin(theta)*cos(psy), sin(phi)*sin(psy)-cos(phi)*sin(theta)*cos(psy)],\
                    [-1*cos(theta)*sin(psy), cos(phi)*cos(psy)-sin(phi)*sin(theta)*sin(psy), sin(phi)*cos(psy)+cos(phi)*sin(theta)*sin(psy)],\
                    [sin(theta), -1*sin(phi)*cos(theta), cos(phi)*cos(theta)]])

    
#    warp_mat = lambda theta, psy, phi : array([[math.cos(theta)*math.cos(psy), math.cos(phi)*math.sin(psy)+math.sin(phi)*math.sin(theta)*math.cos(psy), math.sin(phi)*math.sin(psy)-math.cos(phi)*math.sin(theta)*math.cos(psy)],\
#                    [-1*math.cos(theta)*math.sin(psy), math.cos(phi)*math.cos(psy)-math.sin(phi)*math.sin(theta)*math.sin(psy), math.sin(phi)*math.cos(psy)+math.cos(phi)*math.sin(theta)*math.sin(psy)],\
#                    [math.sin(theta), -1*math.sin(phi)*math.cos(theta), math.cos(phi)*math.cos(theta)]], dtype='float32')

#    R = ((
#        (cos(theta) * cos(psi)),
#        (-cos(phi) * sin(psi) + sin(phi) * sin(theta) * cos(psi)),
#        (sin(phi) * sin(psi) + cos(phi) * sin(theta) * cos(psi)),
#    ), (
#        (cos(theta) * sin(psi)),
#        (cos(phi) * cos(psi) + sin(phi) * sin(theta) * sin(psi)),
#        (-sin(phi) * cos(psi) + cos(phi) * sin(theta) * sin(psi)),
#    ), (
#        (-sin(theta)),
#        (sin(phi) * cos(theta)),
#        (cos(phi) * cos(theta)),
#    ))

#    R = array(R)
    R = warp_mat(theta, psi, phi)
    
    if verbose > 0:
        print("Shape of Rotation Matrix is", shape(R.T))
        print("Shape of vectors to rotate is", shape(v))

    if verbose > 0:
        print("Change in phi is", phi / Degrees)
        print("Change in theta is",theta / Degrees)

    if verbose > 1:
        print(round(R,2))

    try:
        return dot(v,R.T)
        #return dot(R,v)
    except ValueError:
        print("Shape of R is", shape(R), shape(R.T))
        print("Shape of v is", shape(v), v)
        raise ValueError, "dot product not working due to misalignment?!"


def RotateVectors(vectors, rotateto=array([0, 0, 1]), verbose=0):
    '''
    Orient vector(s) to _rotateto_ as z direction
    '''
    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])
    
    return _RotateVector(vectors,
                         phi=0,
                         theta=-theta(rotateto),
                         psi=-phi(rotateto),
                         verbose=verbose)


def _SampledDirection(N, loc, scale, dist, verbose=0):
    '''
    Generate N beams with profile given by the distribution with
    known scale and loc parameters
    '''

    Theta = dist(loc=loc, scale=scale).rvs(N)  # sampled theta
    Phi = random.uniform(-1, 1, N) * pi  # uniform phi

    if verbose > 0:
        print("Theta is", Theta/Degrees)
        print("Phi is", Phi/Degrees)

    X = sin(Theta) * cos(Phi)
    Y = sin(Theta) * sin(Phi)
    Z = cos(Theta)

    newvectors = dstack((X, Y, Z))[0, ...]
    
    if verbose > 0:
        print("Shape of each is", shape(X), shape(Y), shape(Z))
        print("Shape of newvectors is", shape(newvectors))

    return newvectors

def SpecularReflection(olddirection, surfacenormal, ndots, verbose=0):
    '''
    Specular (mirror-like) Reflection
    '''    
    newdirection = olddirection - 2 * ndots[:, newaxis] * surfacenormal

    return newdirection


def LobeReflection(N=1, newdirection=array([0, 0, 1]), stddev=0*Degrees, verbose=0):
    '''
    Gives normal distribution with a standard deviation of 1.3 degrees
    (corresponding to a polished surface - Moses2010)

    orient to specular direction
    '''

    sampledirection = _SampledDirection(N, loc=0, scale=stddev, dist=stats.norm,
                                    verbose=verbose)
    
    orienteddirection = RotateVectors(sampledirection, newdirection, verbose)

    if verbose > 1:
        print("--LobeReflection--")
        print("Sampled direction :", round(sampledirection,2))
        print("Orient vector to :",round(newdirection,2))
        print("Oriented direction :",round(orienteddirection,2))

    if stddev == 0:
        if not allclose(newdirection, orienteddirection):
            print("?!")
        #assert not allclose(newdirection, orienteddirection), "Specular reflections don't match!"


    return orienteddirection


def LambertianReflection(N=1, surfacenormal=array([0, 0, 1]), verbose=0):
    '''
    Gives Lambertian distribution

    orients to surface normal
    '''
    
    class thetadist:
        def __init__(self,loc,scale):
            pass
        def rvs(self,N):
            return arcsin(random.uniform(0, 1, N))

    adirection = _SampledDirection(N, loc=0, scale=0, dist=thetadist,
                             verbose=verbose)

    if verbose > 0:
        print("Shape of lobe reflections is",shape(adirection))

    
    #rotate to -ve of surface normal --> photons orient into the bloody box
    return RotateVectors(adirection, -surfacenormal, verbose)


def IsotropicReflection(N=1, surfacenormal=array([0, 0, 1]), verbose=0):
    '''
    no preferred direction hemispherical emission

    orients to surface normal
    '''
    adirection = _RandomPointsOnASphere(N, hemisphere=True)
    
    return RotateVectors(adirection, surfacenormal, verbose)


def IsotropicSegmentReflection(N=1, surfacenormal=array([0, 0, 1]), 
                    mintheta=0, maxtheta=90*Degrees, fetchall=False, verbose=0):
    '''
    Returns N directions from sphere point picking (Marsaglia 1972) which are inside
    the _mintheta_ and _maxtheta_ allowed polar angles
    '''
    
    while True:
        ListOfDirections = GenerateIsotropicList(N)
        
        mincondition = ListOfDirections[...,2] < cos(mintheta)
        maxcondition = ListOfDirections[...,2] > cos(maxtheta)
            
        try: #matching directions aren't wasted when we generate more
            AllowedDirections = vstack((AllowedDirections,
                                   ListOfDirections[mincondition & maxcondition]))
        except UnboundLocalError:
            AllowedDirections = ListOfDirections[mincondition & maxcondition]
    
        if verbose > 0:
            print("Additional matching is",sum(mincondition & maxcondition))
            print("Total is now",len(AllowedDirections))
            print("Shape of AllowDirections is",shape(AllowedDirections))
    
        if len(AllowedDirections) < N:
            continue
        
        if fetchall:
            return RotateVectors(AllowedDirections, -surfacenormal, verbose)
        else:
            indices = random.choice(range(len(AllowedDirections)),N)
            return RotateVectors(AllowedDirections[indices], -surfacenormal, verbose)

def DirectionVector(theta=0*Degrees, phi=0*Degrees, amplitude=1, verbose=0):             
    '''
    Spherical coordinates (r,theta,phi) --> cartesian coordinates (x,y,z)
    '''
    if issubdtype(type(theta), float) and issubdtype(type(phi), float):
        return (
            array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
        )

    x = amplitude * sin(theta) * cos(phi)
    y = amplitude * sin(theta) * sin(phi)
    if issubdtype(type(theta), float):
        z = amplitude * cos(theta) * ones(shape(x))
    else:
        z = amplitude * cos(theta)
    if verbose > 0:
        print("x", x)
        print("y", y)
        print("z", z)

    return dstack([x, y, z])

def tocube(axis, anum=1):
    axis.set_xlabel("x", size=20)
    axis.set_ylabel("y", size=20)
    axis.set_zlabel("z", size=20)
    axis.set_xlim(-anum, anum)
    axis.set_ylim(-anum, anum)
    axis.set_zlim(-anum, anum)

def _RandomPointsOnASphere(N, hemisphere=False, split=False):
    '''
    Generates random points on a sphere
    or on a hemisphere (default is sphere)
    '''
    Values = []

    while len(Values) < N:
        x1 = 2 * random.random() - 1
        x2 = 2 * random.random() - 1
        
        if hemisphere:
            if (x1 ** 2 + x2 ** 2 < 1) and (x1 ** 2 + x2 ** 2 < 0.5):
                Values.append((x1, x2))
        else:                
            if x1 ** 2 + x2 ** 2 < 1:
                Values.append((x1, x2))

    x1, x2 = zip(*Values)
    x1 = array(x1)
    x2 = array(x2)
    x = 2 * x1 * sqrt(1 - x1 ** 2 - x2 ** 2)
    y = 2 * x2 * sqrt(1 - x1 ** 2 - x2 ** 2)
    z = 1 - 2 * (x1 ** 2 + x2 ** 2)

    return dstack((x, y, z))[0, ...]


def GenerateIsotropicList(N):
    return _RandomPointsOnASphere(N, hemisphere=False)

def IsotropicSource(N, Pos=[0, 0, 0]):
    '''
    Returns a list of initial photons of size N
    direction, position, times
    '''
    return _RandomPointsOnASphere(N), array(list([Pos]) * N), zeros(N)


def TestSource(Pos, aBox):
    
    return aBox.normals, array(list([Pos]) * 6), zeros(6)

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
            print(f, ":", dst)

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
            print(ang, ":", esc)
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

def _firsttrue(param):
    '''
    Returns index of first true in sequential bool array
    '''
    for j,p in enumerate(param):
        if p:
            return j

def _getnewdirection(key, olddirection, ndots, surfacenormal, verbose=0):
    '''    
    Returns newdirection based reflection model chosen by _key_

    newdirections should be oriented correctly here
    '''
    if key == 0: #specular
        return SpecularReflection(olddirection, surfacenormal, ndots, verbose=verbose)    
    elif key == 1: #lobe
        newspeculardirection = SpecularReflection(olddirection, surfacenormal, ndots, verbose=verbose)                
        return vstack([LobeReflection(1, nsd, verbose=verbose) for nsd in newspeculardirection])
    elif key == 2: #backscatter
        return -1*olddirection
    elif key == 3: #lambertian
        return LambertianReflection(len(ndots),surfacenormal)
    elif key == 4: #confined hemisphere
        return IsotropicSegmentReflection(len(ndots),surfacenormal)
    else:
        raise NotImplementedError, "Unknown Reflection type!"
        return 

def UpdateDirection(olddirection, faces, ndots, aBox, verbose=0):
    '''
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    '''
    
    unifiedparameters = zeros((len(faces),4))
    
    for uniqueface in set(faces):
        Condition = (faces == uniqueface)
        unifiedparameters[Condition] = aBox.GetUnified(uniqueface)

    whichreflection = array([_firsttrue(ru < up) for (ru,up) 
                in zip(random.uniform(size=len(faces)), unifiedparameters)])
    

    namedict = {0 : 'Specular', 1: 'Lobe', 2: 'Backscatter', 3: 'Lambertian'}
    if verbose > 0:
        for aref in set(whichreflection):
            print(namedict[aref], len(whichreflection[whichreflection==aref]))
    
    newdirection = zeros(shape(olddirection)) #newdirection
    
    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        for aref in set(whichreflection): #faces x numunified (6 x 4 groups for a cube!)
            Condition = (faces == uniqueface) & (whichreflection == aref)
            if not any(Condition):
                continue
            
            newdirection[Condition,...] = _getnewdirection(aref, olddirection[Condition,...],
                                                       ndots[Condition,...],
                                                       surfacenormal, verbose=verbose)

    
    if verbose > 1:
        print("--Update Direction--")
        for od, nd in zip(olddirection, newdirection):
            print("Old direction :", od)
            print("New direction : ", nd)

             
    return newdirection        
    

def UpdatePosition(oldposition, distanceto, oldtime, directions, aBox, verbose=0):
                   
    '''
    Moves photons to new position
    '''

    newposition = oldposition + distanceto[:, newaxis] * directions
    newtime = oldtime + aBox.n * distanceto / SpeedOfLight

    if verbose > 1:
        print("--Update Position--")
        for np, nt in zip(newposition, newtime):
            print("New position : ", np)
            print("Updated time : ", nt)

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
    maxrepeat = kwargs.get('maxrepeat',10)
    nothingescaped = 0
    
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

        if not any(est):
            nothingescaped += 1

        if verbose > 0:
            print("--Photons Escaped--")
            print(runnum, ": Escaped", sum(est))
            print("\n\n")
        
        ProcessedPhotons += [
            {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
             "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
             "face": fc, "time": atime, "ndots": nds, "photonstatus": "Escaped"}
            for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
            zip(positions[est], directions[est], faces[est], times[est], ndots[est])]
                                  

        directions = directions[est == False]
        positions = positions[est == False]
        times = times[est == False]

        if nothingescaped > maxrepeat:
            if verbose > 0:
                print("No photons escaped in",maxrepeat,"runs, therefore giving up")
            break

    # adds on all remaining 'trapped' photons
    ProcessedPhotons += [
        {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
         "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
         "face": fc, "time": atime, "ndots": nds, "photonstatus": "Trapped"}
        for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
        zip(positions, directions, faces, times, ndots)]
                              

    return ProcessedPhotons
