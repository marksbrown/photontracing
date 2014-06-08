"""
light.py

Author : Mark S. Brown
First Commit : 3rd November 2013

Description : In this module functionality required for ray tracing in geometry
defined in _box_ is contained.

"""
from __future__ import print_function, division
from numpy import array, dot, sin, cos, ones, arccos, cross, mean, std, inner
from numpy import abs, random, zeros, where, sqrt, arcsin
from numpy import shape, arctan2, dstack, newaxis, issubdtype
from numpy import vstack, invert
from pandas import DataFrame
from .const import *


def random_integers(M, N):
    """
    returns N random numbers from the range 0 to M
    :rtype : list
    :param M: max integer
    :param N: number of integers to fetch
    """
    return random.randint(0, M, N)


def dot_python(u, v):
    """
    Pure Python implementation of dot product
    """
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2])


def cross_python(u, v):
    """
    Pure python implementation of cross product
    """
    return u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]


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


def LobeReflection(N, olddirection, surfacenormal, scale=1.3 * Degrees, **kwargs):
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

    dist = kwargs.get("dist", lambda n, scale: scale * random.randn(N))  #defaults to random normal distribution
    verbose = kwargs.get("verbose", 0)

    outgoing_vector = SpecularReflection(olddirection, surfacenormal, dot(olddirection, surfacenormal), verbose)

    if verbose > 0:
        expected_angle = arccos(dot(outgoing_vector, surfacenormal))
        print("Expected angle with surface normal is {0} degrees".format(expected_angle/Degrees))

    t2 = cross(outgoing_vector, olddirection)  #perpendicular to plane
    t3 = cross(t2, outgoing_vector)  #parallel to plane

    amplitudes = dist(N, scale)
    random_numbers = random.rand(N)[..., newaxis]

    A = 1/ sqrt(1 + amplitudes ** 2)[..., newaxis]

    deviation_vector = A * (outgoing_vector + amplitudes[..., newaxis]*(t2*random_numbers + t3*sqrt(1-random_numbers**2)))

    #deviation_vector = 1 / sqrt(1 + amplitudes ** 2)[..., newaxis] * add(outgoing_vector,
    #                                                         [anamplitude * (U * t2 + sqrt(1 - U ** 2) * t3) for
    #                                                          U, anamplitude in zip(random.rand(N), amplitudes)])

    return deviation_vector

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
                indices = random_integers(len(AllowedDirections), N)
            return (
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


def IsotropicSource(N, Pos=(0, 0, 0)):
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
    for uniqueface in set(faces):  #groupby surfacenormal
        surfacenormal = aBox.normals[uniqueface]*-1
        Condition = (faces == uniqueface)

        if not any(Condition):
            continue


        ratio_of_indices = aBox.n / aBox.mat[uniqueface].n
        direction_dot_surfacenormal = dot(Directions[Condition], surfacenormal)
        nds = sqrt(1 - ratio_of_indices ** 2 * (1 - direction_dot_surfacenormal ** 2))

        NewDirections[Condition] = ratio_of_indices * (Directions[Condition] - direction_dot_surfacenormal[..., newaxis] * surfacenormal) + nds[..., newaxis] * surfacenormal

        if verbose > 0:
            print(shape(Directions), shape(NewDirections))
            angles = arccos(inner(Directions, NewDirections))
            print("{} +/- {} Degrees".format(mean(angles), std(angles)))

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
            print(f, ":", dst / mm, "mm", aBox.n * dst / SpeedOfLight / ps, "ps")

    return Faces, DistanceTo, nds


def EscapeStatus(faces, ndots, aBox, reflectivity=True, fresnel=True, surface_layer='inner', verbose=0):
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
    escape_status = angles < CritAngles  # Incident angle less than critical angle?

    if verbose > 0:
        print("--Critical--")
        print("Escaping", sum(escape_status))

    if fresnel:
        Fresnel = aBox.Fresnel(faces, angles)
        ru = random.uniform(size=len(faces))
        escape_status &= (Fresnel < ru)  # Reflection coefficient less than

        if verbose > 0:
            print("--Fresnel--")
            print("Escaping", sum(escape_status))

    if reflectivity:
        Reflectivities = zeros(shape(faces))
        for uniqueface in set(faces):
            Reflectivities[faces == uniqueface] = aBox.Ref(uniqueface, surface_layer)

        ru = random.uniform(size=len(faces))
        escape_status &= (Reflectivities < ru)

        if verbose > 0:
            print("--Reflectivity--")
            print("Reflectivity : Escaped?")
            print("Escaping", sum(escape_status))

    if verbose > 0:
        print("{} photons incident with {} escaping".format(len(escape_status), sum(escape_status)))

    return escape_status  # if true, photon escapes


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
    if verbose > 0:
        reflection = ['Specular', 'Lobe', 'Backscatter', 'Lambertian', 'Segment']
        print("{0} corresponds to {1} reflection".format(key, reflection[key]))

    if key == 0:  # specular
        return SpecularReflection(olddirection, surfacenormal, ndots, verbose=verbose)
    elif key == 1:  # lobe
        return LobeReflection(len(olddirection), olddirection, surfacenormal, scale=mat.lobeangle, verbose=verbose)
    elif key == 2:  # backscatter
        return -1 * olddirection
    elif key == 3:  # lambertian
        return LambertianReflection(len(ndots), surfacenormal)
    elif key == 4:  # confined hemisphere
        return IsotropicSegmentReflection(len(ndots), surfacenormal, mat)
    else:
        raise NotImplementedError("Unknown Reflection type!")


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


def UpdateDirection(olddirection, faces, ndots, aBox, surface_layer='inner', verbose=0):
    """
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    """

    unifiedparameters = zeros((len(faces), 5))

    if verbose > 0:
        print("The unified parameters for the {} surface are {}".format(surface_layer, aBox.GetUnified(0, surface_layer)))

    for uniqueface in set(faces):
        Condition = (faces == uniqueface)
        unifiedparameters[Condition] = aBox.GetUnified(uniqueface, surface_layer)

    whichreflection = array([_firsttrue(ru < up) for (ru, up)
                             in zip(random.uniform(size=len(faces)), unifiedparameters)])

    newdirection = zeros(shape(olddirection))  # newdirection

    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        # faces x numunified (6 x 4 groups for a cube!)
        for aref in set(whichreflection):
            Condition = (faces == uniqueface) & (whichreflection == aref)
            if verbose > 1:
                print("There are {} photons with {} reflection key".format(sum(Condition), aref))
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
            print("prior time was : ", ot / ps, "ps")
            print("Updated time : ", nt / ps, "ps")

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

    enable_snell = kwargs.get('enable_snell', True)
    reflectivity = kwargs.get('reflectivity', True)
    fresnel = kwargs.get('fresnel', True)
    maxrepeat = kwargs.get('maxrepeat', 10)
    specular_only = kwargs.get('specularonly', False)
    nothingescaped = 0

    ProcessedPhotons = []

    ##initial setup
    directions = idir
    positions = ipos
    times = itime

    for runnum in range(runs):
        if verbose > 1:
            print("Onto {} run".format(runnum))

        faces, distanceto, ndots = NearestFace(directions, positions,
                                               aBox, verbose=verbose)

        positions, times = UpdatePosition(positions, distanceto, times,
                                          directions, aBox, verbose=verbose)

        #escape status for a polished uncoated surface --> 
        #aBox.reflectivity is dealt with in the secondary escape status
        escaped_inner = EscapeStatus(faces, ndots, aBox, fresnel=fresnel,
                                     reflectivity=False, verbose=verbose)

        #only update directions of not escaping photons

        if specular_only:
            direction_func = UpdateSpecularDirection
        else:
            direction_func = UpdateDirection

        reflected_from_inner = invert(escaped_inner)

        directions[reflected_from_inner, ...] = direction_func(directions[reflected_from_inner, ...],
                                                               faces[reflected_from_inner, ...],
                                                               ndots[reflected_from_inner, ...],
                                                               aBox, verbose=verbose)

        if hasattr(aBox, 'outer_materials'):
            escaped_outer = EscapeStatus(faces,
                                         ndots,
                                         aBox, fresnel=False,
                                         reflectivity=reflectivity,
                                         surface_layer='outer',
                                         verbose=verbose)

            escaped = escaped_outer & escaped_inner
            reflected_from_outer = invert(escaped_outer) & escaped_inner

            assert not any(reflected_from_outer & reflected_from_inner), "inconsistent definition of reflections"

            if verbose > 0:
                print("{} photons reach the outer surface with {} escaping leading to {} reflected".format(sum(escaped_inner),
                                                                                                           sum(escaped),
                                                                                                           sum(reflected_from_outer)))

            if any(reflected_from_outer):  # update direction of trapped photons from outer surface
                #Photons STILL not escaping gain a new direction!
                directions[reflected_from_outer, ...] = UpdateDirection(directions[reflected_from_outer, ...],
                                                                                faces[reflected_from_outer, ...],
                                                                                ndots[reflected_from_outer, ...],
                                                                                aBox, surface_layer='outer',
                                                                                verbose=verbose)

                ## Bend photons back towards the centre
                if enable_snell:
                    directions[reflected_from_outer, ...] = vectorsnell(directions[reflected_from_outer, ...],
                                                                        faces[reflected_from_outer, ...],
                                                                        aBox, verbose=verbose)

        else:
            escaped = escaped_inner

        if not any(escaped):
            nothingescaped += 1

        if verbose > 0:
            print("--Photons Escaped--")
            print(runnum, ": Escaped", sum(escaped))
            print("\n\n")

        ProcessedPhotons += [
            {"xpos": float(xpos), "ypos": float(ypos), "zpos": float(zpos),
             "xdir": float(xdir), "ydir": float(ydir), "zdir": float(zdir),
             "face": fc, "time": atime, "ndots": nds, "photonstatus": "Escaped"}
            for (xpos, ypos, zpos), (xdir, ydir, zdir), fc, atime, nds in
            zip(positions[escaped], directions[escaped], faces[escaped], times[escaped], ndots[escaped])]

        directions = directions[escaped == False]
        positions = positions[escaped == False]
        times = times[escaped == False]

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
