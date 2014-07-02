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


def random_integers(m, n):
    """
    returns n random numbers from the range 0 to m
    :rtype : list
    :param m: max integer
    :param n: number of integers to fetch
    """
    return random.randint(0, m, n)


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


def _rotate_vector(v, phi=0, theta=0, psi=0):
    """
    rotate vector 'v' using Euler Angles
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


def rotate_vectors(vectors, rotate_to=array([0, 0, 1]), verbose=0):
    """
    Orient vector(s) to _rotateto_ as z direction
    """
    theta = lambda adir: arccos(adir[..., 2])
    phi = lambda adir: arctan2(adir[..., 1], adir[..., 0])

    return _rotate_vector(vectors,
                          phi=0,
                          theta=-theta(rotate_to),
                          psi=-phi(rotate_to))


def _sampled_direction(N, loc, scale, dist, verbose=0):
    """
    Generate N beams with profile given by the distribution with
    known scale and loc parameters
    """

    theta = dist(loc=loc, scale=scale).rvs(N)  # sampled theta
    phi = random.uniform(-1, 1, N) * pi  # uniform phi

    if verbose > 0:
        print("Theta is", theta / Degrees)
        print("Phi is", phi / Degrees)

    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

    newvectors = dstack((x, y, z))[0, ...]

    if verbose > 0:
        print("Shape of each is", shape(x), shape(y), shape(z))
        print("Shape of newvectors is", shape(newvectors))

    return newvectors


def specular_reflection(old_direction, surface_normal, ndots):
    """
    Specular (mirror-like) Reflection
    """
    try:
        new_direction = old_direction - 2 * ndots[:, newaxis] * surface_normal
    except IndexError:
        new_direction = old_direction - 2 * ndots * surface_normal

    return new_direction


def lobe_reflection(N, old_direction, surface_normal, **kwargs):
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

    dist = kwargs.get("dist", lambda n, scale: scale * random.randn(N))  # defaults to random normal distribution
    scale = kwargs.get("scale", 1.3 * Degrees)
    verbose = kwargs.get("verbose", 0)

    outgoing_vector = specular_reflection(old_direction, surface_normal, dot(old_direction, surface_normal))

    t2 = cross(outgoing_vector, old_direction)  # perpendicular to plane
    t3 = cross(t2, outgoing_vector)  # parallel to plane

    amplitudes = dist(N, scale)
    random_numbers = random.rand(N)[..., newaxis]

    A = 1 / sqrt(1 + amplitudes ** 2)[..., newaxis]

    deviation_vector = A * (outgoing_vector + amplitudes[..., newaxis] * (t2 * random_numbers + t3 * sqrt(1 - random_numbers ** 2)))

    return deviation_vector


def lambertian_reflection(N=1, surface_normal=array([0, 0, 1]), verbose=0):
    """
    Gives Lambertian distribution

    orients to surface normal
    """

    class ThetaDist:
        def __init__(self, loc, scale):
            pass

        @staticmethod
        def rvs(N):
            return arcsin(random.uniform(0, 1, N))

    new_direction = _sampled_direction(N, loc=0, scale=0, dist=ThetaDist,
                                   verbose=verbose)

    # rotate to -ve of surface normal --> photons orient into the bloody box
    return rotate_vectors(new_direction, -surface_normal, verbose)


def isotropic_reflection(N=1, surface_normal=array([0, 0, 1]), verbose=0):
    """
    no preferred direction hemispherical emission

    orients to surface normal
    """
    new_direction = _random_points_on_a_sphere(N, hemisphere=True)

    return rotate_vectors(new_direction, -surface_normal, verbose)


def theta_segment_reflection(N=1, surface_normal=array([0, 0, 1]), mat=None,
                               fetchall=False, verbose=0):
    """
    Returns N directions from sphere point picking (Marsaglia 1972)
    """

    while True:
        potential_directions = generate_isotropic_source(N)

        min_theta_bool = potential_directions[..., 2] < cos(mat.mintheta)
        max_theta_bool = potential_directions[..., 2] > cos(mat.maxtheta)

        try:  # matching directions aren't wasted when we generate more
            new_directions = vstack((new_directions, potential_directions[min_theta_bool & max_theta_bool]))
        except UnboundLocalError:
            new_directions = potential_directions[min_theta_bool & max_theta_bool]

        if verbose > 0:
            print("Additional matching is", sum(min_theta_bool & max_theta_bool))
            print("Total is now", len(new_directions))
            print("Shape of AllowDirections is", shape(new_directions))

        if len(new_directions) < N:
            continue

        if fetchall:
            return rotate_vectors(new_directions, -surface_normal, verbose)
        else:   # returns 'N' valid new directions only
            try:
                indices = random.choice(range(len(new_directions)), N)
            except AttributeError:
                indices = random_integers(len(new_directions), N)
            return rotate_vectors(new_directions[indices], -surface_normal, verbose)


def theta_phi_segment_reflection(N, surface_normal=array([0, 0, 1]), mat=None, fetchall=False, verbose=0):
    """
    Returns N directions from sphere point picking within theta and phi ranges only
    """
    #theta=(0, pi/6), phi=None, verbose=0, max_iterations=100

    if mat.maxphi is None:
        if verbose > 0:
            print("No phi range defined, skipping to default theta_segment_reflection")
        return theta_segment_reflection(N, surface_normal, mat, fetchall, verbose)

    min_theta = mat.mintheta
    max_theta = mat.maxtheta

    min_phi = mat.minphi
    max_phi = mat.maxphi

    while True:

        potential_directions = generate_isotropic_source(N, 'all')

        phi = arctan2(potential_directions[..., 1], potential_directions[..., 0])
        phi[phi < 0] += 2*pi

        min_phi_condition = phi > min_phi
        max_phi_condition = phi < max_phi

        min_theta_condition = potential_directions[..., 2] < cos(min_theta)
        max_theta_condition = potential_directions[..., 2] > cos(max_theta)

        min_condition = min_phi_condition & min_theta_condition
        max_condition = max_phi_condition & max_theta_condition

        try:  # matching directions aren't wasted when we generate more
            new_directions = vstack((new_directions,
                                        potential_directions[min_condition & max_condition]))
        except UnboundLocalError:
            new_directions = potential_directions[min_condition & max_condition]

        if verbose > 0:
            print("Additional matching is", sum(min_condition & max_condition))
            print("Total is now", len(new_directions))
            print("Shape of AllowDirections is", shape(new_directions))

        if len(new_directions) < N:
            continue

        if fetchall:
            return rotate_vectors(new_directions, -surface_normal, verbose)
        else:   # returns 'N' valid new directions only
            try:
                indices = random.choice(range(len(new_directions)), N)
            except AttributeError:
                indices = random_integers(len(new_directions), N)
            return rotate_vectors(new_directions[indices], -surface_normal, verbose)

    return None


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


def radial_direction_vector(theta=0, phi=0, amplitude=1):
    """
    returns r unit vector in cartesian coordinates
    """

    return array((amplitude * sin(theta) * cos(phi),
                  amplitude * sin(theta) * sin(phi),
                  amplitude * ones(shape(phi)) * cos(theta))).T


def _random_points_on_a_sphere(N, hemisphere=False):
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


def generate_isotropic_source(N):
    return _random_points_on_a_sphere(N, hemisphere=False)



def snell_vectorised(Directions, faces, aBox, verbose=0):
    """
    Vectorised Snell Operation
    """
    NewDirections = ones(shape(Directions))
    for uniqueface in set(faces):  # groupby surfacenormal
        surface_normal = aBox.normals[uniqueface] * -1
        Condition = (faces == uniqueface)

        if not any(Condition):
            continue

        ratio_of_indices = aBox.n / aBox.mat[uniqueface].n
        direction_dot_surfacenormal = dot(Directions[Condition], surface_normal)
        nds = sqrt(1 - ratio_of_indices ** 2 * (1 - direction_dot_surfacenormal ** 2))

        NewDirections[Condition] = ratio_of_indices * (Directions[Condition] - direction_dot_surfacenormal[..., newaxis] * surface_normal) + nds[..., newaxis] * surface_normal

        if verbose > 0:
            print(shape(Directions), shape(NewDirections))
            angles = arccos(inner(Directions, NewDirections))
            print("{} +/- {} Degrees".format(mean(angles), std(angles)))

    return NewDirections


def nearest_face(Directions, Positions, aBox, verbose=0, threshold=1e-15):
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


def face_escape_status(faces, ndots, aBox, surface_layer='inner', verbose=0):
    """
    Photons arriving at a surface will change status to 'trapped','escaped'
    or 'absorbed' based on order of events
    1st test : critical angle (trapped if angle is within)
    2nd test : fresnel reflection
    3rd test : reflectivity parameter - this WILL override everything else
    """

    escape_status = ones(shape(faces), dtype=bool)


    for unique_face in set(faces):
        condition = faces == unique_face
        interactions = aBox.face_interactions_enabled(unique_face, surface_layer)
        if not any(interactions):
            continue
        critical, fresnel, reflectivity = interactions

        angles = arccos(ndots[condition])

        if critical:
            critical_angle_of_face = aBox.get_critical_angle(unique_face)
            escape_status[condition] = angles < critical_angle_of_face  # Incident angle less than critical angle?

        if fresnel:
            fresnel_of_face = aBox.fresnel_reflectance(faces[condition], angles)
            escape_status[condition] &= (fresnel_of_face < random.uniform(size=sum(condition)))  # fresnel < random?

        if reflectivity:
            reflectivity_of_face = aBox.get_reflectivity(unique_face, surface_layer)
            escape_status[condition] &= (reflectivity_of_face < random.uniform(size=sum(condition)))
    
    for unique_face in set(faces):
        if verbose > 0:
            print("{} photons incident with {} escaping from face {}".format(
                len(escape_status[faces == unique_face]), sum(escape_status[faces == unique_face]), unique_face))

    return escape_status  # if true, photon escapes


def _first_true(param):
    """
    Returns index of first true in sequential bool array
    """
    for j, p in enumerate(param):
        if p:
            return j


def _get_new_direction(key, old_direction, ndots, surface_normal, mat, verbose=0):
    """
    Returns newdirection based reflection model chosen by _key_

    newdirections should be oriented correctly here
    """
    if verbose > 1:
        reflection = ['Specular', 'Lobe', 'Backscatter', 'Lambertian', 'Segment']
        print("{0} corresponds to {1} reflection".format(key, reflection[key]))

    if key == 0:  # specular
        return specular_reflection(old_direction, surface_normal, ndots)
    elif key == 1:  # lobe
        return lobe_reflection(len(old_direction), old_direction, surface_normal, scale=mat.lobeangle, verbose=verbose)
    elif key == 2:  # backscatter
        return -1 * old_direction
    elif key == 3:  # lambertian
        return lambertian_reflection(len(ndots), surface_normal)
    elif key == 4:  # confined hemisphere
        return theta_phi_segment_reflection(len(ndots), surface_normal, mat)
    else:
        raise NotImplementedError("Unknown Reflection type!")


def update_specular_direction(old_direction, faces, ndots, aBox, specular_only=True, verbose=0):
    """
    Updates directions using the specular model only
    """

    new_direction = zeros(shape(old_direction))  # newdirection

    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        Condition = (faces == uniqueface)
        if not any(Condition):
            continue

        new_direction[Condition, ...] = _get_new_direction(
            0, old_direction[Condition, ...],
            ndots[Condition, ...],
            surfacenormal, aBox.mat[uniqueface], verbose=verbose)

    return new_direction


def update_direction(old_direction, faces, ndots, aBox, surface_layer='inner', specular_only=False, verbose=0):
    """
    Calculates new direction for a given photon at a given face for a set
    of UNIFIED parameters
    """
    newdirection = zeros(shape(old_direction))  # newdirection

    unifiedparameters = zeros((len(faces), 5))  #  TODO replace 5 with number fetched from aBox.mat class

    if verbose > 1:
        print("The unified parameters for the {} surface are {}".format(surface_layer, aBox.get_surface_parameters(0, surface_layer)))

    for uniqueface in set(faces):
        Condition = (faces == uniqueface)
        unifiedparameters[Condition] = aBox.get_surface_parameters(uniqueface, surface_layer)

    if specular_only:
        which_reflection = (0,)  # specular reflection only

    else:
        which_reflection = array([_first_true(ru < up) for (ru, up)
                                 in zip(random.uniform(size=len(faces)), unifiedparameters)])


    for uniqueface in set(faces):
        surfacenormal = aBox.normals[uniqueface]
        # faces x numunified (6 x 4 groups for a cube!)
        for aref in set(which_reflection):
            if specular_only:
                Condition = faces == uniqueface
            else:
                Condition = (faces == uniqueface) & (which_reflection == aref)

            if not any(Condition):
                continue

            newdirection[Condition, ...] = _get_new_direction(
                aref, old_direction[Condition, ...],
                ndots[Condition, ...],
                surfacenormal, aBox.mat[uniqueface], verbose=verbose)

    if verbose > 1:
        print("--Update Direction--")
        for od, nd in zip(old_direction, newdirection):
            print("Old direction :", od)
            print("New direction : ", nd)

    return newdirection


def update_position(old_position, distanceto, old_time, directions, aBox, verbose=0):
    """
    Moves photons to new position
    """

    new_position = old_position + distanceto[:, newaxis] * directions
    new_time = old_time + aBox.n * distanceto / SpeedOfLight

    if verbose > 1:
        print("--Update Position--")
        for np, ot, nt in zip(new_position, old_time, new_time):
            #print("New position : ", np)
            print("prior time was : ", ot / ps, "ps")
            print("Updated time : ", nt / ps, "ps")

    return new_position, new_time

def step(directions, positions, times, aBox, **kwargs):
    """
    Move photons to next position
    """
    enable_snell = kwargs.get('enable_snell', True)
    specular_only = kwargs.get('specularonly', False)
    verbose = kwargs.get('verbose', 0)


    faces, distanceto, ndots = nearest_face(directions, positions,
                                            aBox, verbose=verbose)

    positions, times = update_position(positions, distanceto, times,
                                      directions, aBox, verbose=verbose)

    #escape status for a polished uncoated surface -->
    #aBox.reflectivity is dealt with in the secondary escape status
    escaped_inner = face_escape_status(faces, ndots, aBox, verbose=verbose)

    #only update directions of not escaping photons

    reflected_from_inner = invert(escaped_inner)

    directions[reflected_from_inner, ...] = update_direction(directions[reflected_from_inner, ...],
                                                           faces[reflected_from_inner, ...],
                                                           ndots[reflected_from_inner, ...],
                                                           aBox,
                                                           specular_only=specular_only,
                                                           verbose=verbose)

    if hasattr(aBox, 'outer_mat'):
        escaped_outer = face_escape_status(faces,
                                     ndots, aBox, verbose=verbose)

        escaped = escaped_outer & escaped_inner
        reflected_from_outer = invert(escaped_outer) & escaped_inner

        assert not any(reflected_from_outer & reflected_from_inner), "inconsistent definition of reflections"

        if verbose > 0:
            print("{} photons reach the outer surface with {} escaping leading to {} reflected".format(
                sum(escaped_inner),
                sum(escaped),
                sum(reflected_from_outer)))

        if any(reflected_from_outer):  # update direction of trapped photons from outer surface
            #Photons STILL not escaping gain a new direction!
            directions[reflected_from_outer, ...] = update_direction(directions[reflected_from_outer, ...],
                                                                    faces[reflected_from_outer, ...],
                                                                    ndots[reflected_from_outer, ...],
                                                                    aBox, surface_layer='outer',
                                                                    verbose=verbose)

            ## Bend photons back towards the centre
            if enable_snell:
                directions[reflected_from_outer, ...] = snell_vectorised(directions[reflected_from_outer, ...],
                                                                    faces[reflected_from_outer, ...],
                                                                    aBox, verbose=verbose)

    else:
        escaped = escaped_inner

    return directions, positions, times, ndots, faces, escaped





class PhotonTrace():
    """
    Creates a simulation that will run _num_of_photons_ within the _geometry_
    """

    def __init__(self, num_of_photons, geometry, **kwargs):

        if 'positions' in kwargs:
            self.positions = kwargs.get('positions')
        elif 'position' in kwargs:
            self.positions = array(list([kwargs.get('position')]) * num_of_photons)
        else:
            self.positions = array(list([0,0,0]) * num_of_photons)

        self.directions = _random_points_on_a_sphere(num_of_photons)
        self.times = kwargs.get('times', zeros(num_of_photons))  # defaults to zero for each photon
        self.aBox = geometry

        self.faces = zeros(num_of_photons)
        self.ndots = zeros(num_of_photons)
        self.photon_status = ones(num_of_photons, dtype=bool)  # 1 corresponds to trapped, 0 to escaped
        self.energy = kwargs.get('energy', ones(num_of_photons)/num_of_photons)
        self.photons_escaped_last = 0
        #df.time /= ps


    def describe_data(self):
        """
        print output describing the system
        times in ps

        """

        out_string = """
        Simulation holds {} photons of which {} have escaped. This corresponds to {:2.2f} % escaped
        """

        num_of_photons = len(self.photon_status)
        num_of_escaped = sum(invert(self.photon_status))
        print(out_string.format(num_of_photons, num_of_escaped, num_of_escaped/num_of_photons*1e2))


    def fetch_stats(self):
        return self.photons_escaped_last

    def fetch_data(self):
        """
        returns list of dictionaries of photons which have escaped
        """

        return [{'xpos': apos[0], 'ypos': apos[1], 'zpos': apos[2],
                'xdir': adir[0],  'ydir': adir[1], 'zdir': adir[2],
                'time': atime,    'face': face,    'angle': angle, 'energy': energy,
                'photonstatus': status} for apos, adir, atime, face, angle, energy, status in zip(self.positions,
                                                                                      self.directions,
                                                                                      self.times/ps,
                                                                                      self.faces,
                                                                                      arccos(self.ndots),
                                                                                      self.energy,
                                                                                      self.photon_status)]

    def run_gen(self, runs=10, **kwargs):
        """
        Data is yielded every iteration allowing creation of animation
        """

        fetch = kwargs.get('fetch', self.fetch_data)

        yield fetch()

        for run_num in range(runs):
            self.run(**kwargs)
            yield fetch()


    def run(self, runs=1, **kwargs):
        """
        Given known initial conditions we propagate photons through the chosen
        geometry

        idir : initial directions
        ipos : initial positions
        itime : initial times
        aBox : Geometry object (see box.py)
        runs : number of `bounces' to attempt
        """

        max_repeat = kwargs.get('maxrepeat', 10)
        verbose = kwargs.get('verbose', 0)

        nothing_escaped_counter = 0  # if above max_repeat the simulation is halted

        for run_num in range(runs):
            photons_trapped = sum(self.photon_status)  # True are trapped

            if verbose > 0:
                print("\nOnto {} run with {} photons ({}/{})".format(run_num,
                                                                     len(self.photon_status),
                                                                     sum(self.photon_status),
                                                                     sum(invert(self.photon_status))))



            self.directions[self.photon_status], self.positions[self.photon_status], \
            self.times[self.photon_status], self.ndots[self.photon_status], self.faces[self.photon_status], \
            escaped_photons = step(self.directions[self.photon_status], self.positions[self.photon_status],
                                                            self.times[self.photon_status], self.aBox, **kwargs)


            ## Locations in lists where there is a change from trapped to escaped are described by the XOR below
            self.photons_escaped_last = sum(self.photon_status[self.photon_status] ^ invert(escaped_photons))


            self.photon_status[self.photon_status] = invert(escaped_photons)

            #self.photon_status[escaped_photons]= False  # Escaped photons are no longer bothered

            if self.photons_escaped_last == 0:  # No additional photons have escaped
                nothing_escaped_counter += 1

                if nothing_escaped_counter > max_repeat:
                    print("Giving up on run {} after {} runs with no escape".format(run_num, max_repeat))
                    break


