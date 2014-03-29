"""
light.py

Author : Mark S. Brown
First Commit : 3rd November 2013

Description : In this module functionality required for ray tracing in geometry
defined in _box_ is contained.

"""
from __future__ import print_function, division
from numpy import array, dot, sin, cos, ones, arccos, cross
from numpy import abs, random, zeros, where, sqrt, arcsin
from numpy import shape, arctan2, dstack, newaxis, issubdtype
from numpy import vstack, invert, add

from pandas import DataFrame
from .const import *

def random_integer(M,N):
    """
    returns N random numbers from the range 0 to M
    """
    return random.randint(0,M,N)

def dot_python(u, v):
    """
    Pure Python implementation of dot product
    """
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])

def cross_python(u, v):
    """
    Pure python implementation of cross product
    """
    return u[1]*v[2]-u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]


def _rotate_vector(v, phi=0, theta=0, psi=0, verbose=0):
    """
    rotate vector 'v' using Euler Angles
    :rtype : list or array
    :param v: (...,3) vector(s)
    :param phi: yaw
    :param theta: pitch
    :param psi: roll?
    :param verbose: verbosity control
    """

    # http://stackoverflow.com/questions/19470955/warping-an-image-using-roll-pitch-and-yaw
    warp_mat = lambda theta, psy, phi: array(
        [[cos(theta) * cos(psy), cos(phi) * sin(psy) 
            + sin(phi) * sin(theta) * cos(psy), 
            sin(phi) * sin(psy) - cos(phi) * sin(theta) * cos(psy)],
         [-1 * cos(theta) * sin(psy), cos(phi) * cos(psy) 
            - sin(phi) * sin(theta) * sin(psy),
            sin(phi) * cos(psy) + cos(phi) * sin(theta) * sin(psy)],
         [sin(theta), -1 * sin(phi) * cos(theta), cos(phi) * cos(theta)]])

    R = warp_mat(theta, psi, phi)

    try:
        return dot(v, R.T)
    except ValueError:
        print("Shape of R is", shape(R), shape(R.T))
        print("Shape of v is", shape(v), v)
        raise ValueError("dot product not working due to misalignment!")


def RotateVectors(vectors, rotateto=array([0, 0, 1]), verbose=0):
    """
    Orient vector(s) to _rotateto_ as z direction
    """
    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])

    return _rotate_vector(vectors,
                         phi=0,
                         theta=-theta(rotateto),
                         psi=-phi(rotateto),
                         verbose=verbose)


def _SampledDirection(N, loc, scale, dist, verbose=0):
    """
    Generate N beams with profile given by the distribution with
    known scale and loc parameters
    """

    Theta = dist(loc=loc, scale=scale).rvs(N)  # sampled theta
    Phi = random.uniform(-1, 1, N) * pi  # uniform phi

    if verbose > 0:
        print("Theta is", Theta / Degrees)
        print("Phi is", Phi / Degrees)

    X = sin(Theta) * cos(Phi)
    Y = sin(Theta) * sin(Phi)
    Z = cos(Theta)

    newvectors = dstack((X, Y, Z))[0, ...]

    if verbose > 0:
        print("Shape of each is", shape(X), shape(Y), shape(Z))
        print("Shape of newvectors is", shape(newvectors))

    return newvectors


def SpecularReflection(olddirection, surfacenormal, ndots, verbose=0):
    """
    Specular (mirror-like) Reflection
    """
    try:
        newdirection = olddirection - 2 * ndots[:, newaxis] * surfacenormal
    except IndexError:
        newdirection = olddirection - 2 * ndots * surfacenormal

    return newdirection

def LobeReflection(N, olddirection, surfacenormal, scale=1.3*Degrees, **kwargs):
    """
    Deviation from specular reflection by a chosen amount (defaults to normal)

    --args--
    N : number of photons to produce
    incoming_vector : vector from incident photon
    surface normal : surface normal of surface
    scale : size parameter for deviation vector

    --kwargs--
    dist : distribution function of form f(N, scale) which returns a series of amplitudes
    verbose : verbosity control
    """

    dist = kwargs.get("dist", lambda n, scale : scale*random.randn(N)) #defaults to random normal distribution
    verbose = kwargs.get("verbose", 0)

    outgoing_vector = SpecularReflection(olddirection, surfacenormal, dot(olddirection, surfacenormal), verbose)

    t2 = cross(outgoing_vector, olddirection) #perpendicular to plane
    t3 = cross(t2, outgoing_vector) #parallel to plane

    amplitudes = dist(N, scale)
    return 1/sqrt(1+amplitudes**2)[...,newaxis] * add(outgoing_vector, [anamplitude*(U*t2+sqrt(1-U**2)*t3) for U, anamplitude in zip(random.rand(N), amplitudes)])


def LambertianReflection(N=1, surfacenormal=array([0, 0, 1]), verbose=0):
    """
    Gives Lambertian distribution

    orients to surface normal
    """

    class thetadist:

        def __init__(self, loc, scale):
            pass

        def rvs(self, N):
            return arcsin(random.uniform(0, 1, N))

    adirection = _SampledDirection(N, loc=0, scale=0, dist=thetadist,
                                   verbose=verbose)

    if verbose > 0:
        print("Shape of lobe reflections is", shape(adirection))

    # rotate to -ve of surface normal --> photons orient into the bloody box
    return RotateVectors(adirection, -surfacenormal, verbose)


def IsotropicReflection(N=1, surfacenormal=array([0, 0, 1]), verbose=0):
    """
    no preferred direction hemispherical emission

    orients to surface normal
    """
    adirection = _RandomPointsOnASphere(N, hemisphere=True)

    return RotateVectors(adirection, -surfacenormal, verbose)


def IsotropicSegmentReflection(N=1, surfacenormal=array([0, 0, 1]), mat=None,
                               fetchall=False, verbose=0):
    """
    Returns N directions from sphere point picking (Marsaglia 1972)
    """

    while True:
        ListOfDirections = GenerateIsotropicList(N)

        mincondition = ListOfDirections[..., 2] < cos(mat.mintheta)
        maxcondition = ListOfDirections[..., 2] > cos(mat.maxtheta)

        try:  # matching directions aren't wasted when we generate more
            AllowedDirections = vstack((AllowedDirections,
                                        ListOfDirections[mincondition & maxcondition]))
        except UnboundLocalError:
            AllowedDirections = ListOfDirections[mincondition & maxcondition]

        if verbose > 0:
            print("Additional matching is", sum(mincondition & maxcondition))
            print("Total is now", len(AllowedDirections))
            print("Shape of AllowDirections is", shape(AllowedDirections))

        if len(AllowedDirections) < N:
            continue

        if fetchall:
            return RotateVectors(AllowedDirections, -surfacenormal, verbose)
        else:
            try:
                indices = random.choice(range(len(AllowedDirections)), N)
            except AttributeError:
                indices = random_integer(len(AllowedDirections), N)
            return  (
                    RotateVectors(
                    AllowedDirections[indices],
                    -surfacenormal,
                    verbose)
                    )


def DirectionVector(theta=0 * Degrees, phi=0 * Degrees, amplitude=1, verbose=0):
    """
    Spherical coordinates (r,theta,phi) --> cartesian coordinates (x,y,z)
    """
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

def spherical_unit_vectors(theta=0, phi=0):
    """
    Spherical unit vectors as cartesian unit vectors
    """
    r = array((sin(theta) * cos(phi),
               sin(theta) * sin(phi),
               ones(shape(phi)) * cos(theta))).T
    th = array((cos(theta) * cos(phi),
                cos(theta) * sin(phi),
                -1 * ones(shape(phi)) * sin(theta))).T
    ph = array((-1 * ones(shape(theta)) * sin(phi),
                ones(shape(theta)) * cos(phi),
                zeros(shape(theta)) * zeros(shape(phi)))).T

    return r, th, ph


def radial_direction_vector(theta=0, phi=0, amplitude=1, verbose=0):
    """
    returns r unit vector in cartesian coordinates
    """

    return array((amplitude * sin(theta) * cos(phi),
                  amplitude * sin(theta) * sin(phi),
                  amplitude * ones(shape(phi)) * cos(theta))).T

def _RandomPointsOnASphere(N, hemisphere=False, split=False):
    """
    Generates random points on a sphere
    or on a hemisphere (default is sphere)
    """
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
    """
    Returns a list of initial photons of size N
    direction, position, times
    """
    return _RandomPointsOnASphere(N), array(list([Pos]) * N), zeros(N)

def TestSource(Pos, aBox):

    return aBox.normals, array(list([Pos]) * 6), zeros(6)

def vectorsnell(Directions, faces, aBox, verbose=0):
    """
    Vectorised Snell Operation
    """
    NewDirections = ones(shape(Directions))
    for uniqueface in set(faces): #groupby surfacenormal
        surfacenormal = aBox.normals[uniqueface]
        Condition = (faces == uniqueface)
        if not any(Condition):
            continue

        r = aBox.n / aBox.mat[uniqueface].n

        a = dot(Directions[Condition], surfacenormal)
    
        nds = sqrt(1 - r**2*(1-a**2))
    
        NewDirections[Condition] = r*(n-a*sn)+nds*sn

    return NewDirections

def NearestFace(Directions, Positions, aBox, verbose=0, threshold=1e-15):
    """
    returns the distance to the nearest face, the nearest face index and
    the angle with respect to the surface normal
    """

    nds = zeros(len(Directions))
    Faces = zeros(len(Directions))
    DistanceTo = ones(len(Directions)) / threshold

    # iterates over each face
    for (i, sn), sp in zip(enumerate(aBox.normals), aBox.points):
        ndots = dot(Directions, sn)
        dmin = dot(Positions, sn)
        dmin -= dot_python(sn, sp)
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
            print(f, ":", dst/mm,"mm", aBox.n * dst / SpeedOfLight / ps,"ps")

    return Faces, DistanceTo, nds


def EscapeStatus(faces, ndots, aBox, reflectivity=True, fresnel=True, verbose=0):
    """
    Photons arriving at a surface will change status to 'trapped','escaped'
    or 'absorbed' based on order of events
    1st test : critical angle (trapped if angle is within)
    2nd test : fresnel reflection
    3rd test : reflectivity parameter - this WILL override everything else
    """

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
    """
    Returns index of first true in sequential bool array
    """
    for j, p in enumerate(param):
        if p:
            return j


def _getnewdirection(key, olddirection, ndots, surfacenormal, mat, verbose=0):
    """
    Returns newdirection based reflection model chosen by _key_

    newdirections should be oriented correctly here
    """
    if key == 0:  # specular
        return (
            SpecularReflection(
                olddirection,
                surfacenormal,
                ndots,
                verbose=verbose)
        )
    elif key == 1:  # lobe
        newspeculardirection = SpecularReflection(
            olddirection,
            surfacenormal,
            ndots,
            verbose=verbose)
        return (
            vstack([LobeReflection(1, nsd, stddev=mat.lobeangle, verbose=verbose)
                   for nsd in newspeculardirection])
        )
    elif key == 2:  # backscatter
        return -1 * olddirection
    elif key == 3:  # lambertian
        return LambertianReflection(len(ndots), surfacenormal)
    elif key == 4:  # confined hemisphere
        return IsotropicSegmentReflection(len(ndots), surfacenormal, mat)
    else:
        raise NotImplementedError("Unknown Reflection type!")
        return


def UpdateSpecularDirection(olddirection, faces, ndots, aBox, verbose=0):
    """
    Updates directions using the specular model only
    """

    newdirection = zeros(shape(olddirection))  # newdirection

    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        Condition = (faces == uniqueface)
        if not any(Condition):
            continue

        newdirection[Condition, ...] = _getnewdirection(
            0, olddirection[Condition, ...],
            ndots[Condition, ...],
            surfacenormal, aBox.mat[uniqueface], verbose=verbose)

    return newdirection

def UpdateDirection(olddirection, faces, ndots, aBox, verbose=0):
    """
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    """

    unifiedparameters = zeros((len(faces), 5))

    for uniqueface in set(faces):
        Condition = (faces == uniqueface)
        unifiedparameters[Condition] = aBox.GetUnified(uniqueface)

    whichreflection = array([_firsttrue(ru < up) for (ru, up)
                             in zip(random.uniform(size=len(faces)), unifiedparameters)])

    namedict = {0: 'Specular', 1: 'Lobe', 2: 'Backscatter', 3: 'Lambertian', 4: 'Segment'}
    if verbose > 0:
        for aref in set(whichreflection):
            print(namedict[aref], len(whichreflection[whichreflection == aref]))

    newdirection = zeros(shape(olddirection))  # newdirection

    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        # faces x numunified (6 x 4 groups for a cube!)
        for aref in set(whichreflection):
            Condition = (faces == uniqueface) & (whichreflection == aref)
            if not any(Condition):
                continue

            newdirection[Condition, ...] = _getnewdirection(
                aref, olddirection[Condition, ...],
                ndots[Condition, ...],
                surfacenormal, aBox.mat[uniqueface], verbose=verbose)

    if verbose > 1:
        print("--Update Direction--")
        for od, nd in zip(olddirection, newdirection):
            print("Old direction :", od)
            print("New direction : ", nd)

    return newdirection


def UpdatePosition(oldposition, distanceto,
                   oldtime, directions, aBox, verbose=0):
    """
    Moves photons to new position
    """

    newposition = oldposition + distanceto[:, newaxis] * directions
    newtime = oldtime + aBox.n * distanceto / SpeedOfLight

    if verbose > 1:
        print("--Update Position--")
        for np, ot, nt in zip(newposition, oldtime, newtime):
            #print("New position : ", np)
            print("prior time was : ", ot/ps,"ps")
            print("Updated time : ", nt/ps,"ps")
    
    return newposition, newtime


def ToDataFrame(directions, positions, times, faces):

    adict = [{"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
             "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
              "face": fc, "time": atime}
             for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime in
             zip(positions, directions, faces, times)]

    return DataFrame(adict)


def LightinaBox(idir, ipos, itime, aBox, runs=1, verbose=0, **kwargs):
    """
    Given known initial conditions we propagate photons through the chosen
    geometry

    idir : initial directions
    ipos : initial positions
    itime : initial times
    aBox : Geometry object (see box.py)
    runs : number of `bounces' to attempt
    """

    reflectivity = kwargs.get('reflectivity', True)
    fresnel = kwargs.get('fresnel', True)
    maxrepeat = kwargs.get('maxrepeat', 10)
    outersurface = kwargs.get('outersurface',False)
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
        
        #escape status for a polished uncoated surface --> 
        #aBox.reflectivity is dealt with in the secondary escape status
        est = EscapeStatus(faces, ndots, aBox, fresnel=fresnel,
                           reflectivity=False, verbose=verbose)

        #only update directions of not escaping photons
        directions[invert(est),...] = UpdateSpecularDirection(directions[invert(est),...],
                                    faces[invert(est),...], 
                                    ndots[invert(est),...],
                                     aBox, verbose=verbose)
                                     
        #don't touch invert(est) photons anymore
                

        if outersurface: #Considers a secondary outer surface (Fresnel is dealt with before!)
            esc = ones(shape(faces),dtype=bool) #assume everything escapes
            esc[est,...] = EscapeStatus(faces[est,...], 
                                        ndots[est,...], 
                                        aBox, fresnel=False,
                                        reflectivity=reflectivity, verbose=verbose)

            #if any photons are trapped            
            #print(sum(invert(esc)),"photons trapped!")
            if any(invert(esc)): #no photons are kept
                #Photons STILL not escaping gain a new direction!
                directions[invert(esc),...] = UpdateDirection(directions[invert(esc),...],
                                        faces[invert(esc),...], 
                                        ndots[invert(esc),...],
                                         aBox, verbose=verbose)

                #Bend photons back towards the centre
                #directions[invert(esc),...] = vectorsnell(directions[invert(esc),...], faces[est,...], n1, n2)

            est = est & esc #proper measure of what's what
            
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
                print(
                    "No photons escaped in",
                    maxrepeat,
                    "runs, therefore giving up")
            break

    # adds on all remaining 'trapped' photons
    ProcessedPhotons += [
        {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
         "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
         "face": fc, "time": atime, "ndots": nds, "photonstatus": "Trapped"}
        for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
        zip(positions, directions, faces, times, ndots)]

    return ProcessedPhotons
