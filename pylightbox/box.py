"""
box.py

Author : Mark S. Brown
First Commit : 3rd November 2013

Description : In this module the _box_ class is defined along with associated
functions. This will be used with the _light_ module to perform ray tracing
in geometries defined below. Several examples are given in this file.
"""

from __future__ import print_function, division
from itertools import combinations

from numpy import arcsin, sin, dot, subtract, linalg, cos, isnan
from numpy import tan, zeros, shape

from .generic import *


class Box():  # Box Properties
    """
    Six sided irregular box class

    n : Refractive index (const with wavelength)
    cn : Corners (used to draw frame only)
    sn : surface normal
    sp : paired surface points for each surface normal
    faces : Dict of numbers for labelling each face
    ci : dict of coupling refractive indices for each face
    materials : surface material objects

    Note : each surfacenormal and surfacepoint must define a unique face.
    This system isn't strictly speaking, limited to 6 faces
    """

    def __init__(self, n, cn, sn, sp, faces, materials, name="", **kwargs):
        self.n = n  # refractive index of box
        self.corners = cn  # corners (for drawing frame only)

        # Paired data defining surface faces
        self.normals = array(sn)  # surface normals
        self.points = array(sp)  # surface points
        self.name = name  # name of box
        self.face = faces  # names of each face
        self.mat = materials

        if 'outer_mat' in kwargs:
            self.outer_mat = kwargs['outer_mat']

    def __repr__(self):
        return self.name

    def face_interactions_enabled(self, face, surface_layer='inner'):
        """

        """

        if surface_layer == 'inner':
            return self.mat[face].critical, self.mat[face].fresnel, self.mat[face].ref
        elif surface_layer == 'outer':
            return self.outer_mat[face].critical, self.outer_mat[face].fresnel, self.outer_mat[face].ref



    def get_surface_parameters(self, face, surface_layer='inner'):
        """
        Retrieves unified parameters for a chosen face of form
        [specular,lobe,backscatter,lambertian]
        """

        if surface_layer == 'inner':
            return self.mat[face].surface
        elif surface_layer == 'outer':
            return self.outer_mat[face].surface

    def get_critical_angle(self, face, surface_layer='inner'):
        """
        Returns the critical angle at facet
        """
        if surface_layer == 'inner':
            return arcsin(self.mat[face].n / self.n)
        elif surface_layer == 'outer':
            return arcsin(self.outer_mat[face].n / self.mat[face].n)

    def get_reflectivity(self, face, surface_layer='inner'):

        if surface_layer == 'inner':
            return self.mat[face].reflectivity
        elif surface_layer == 'outer':
            return self.outer_mat[face].reflectivity

    def get_surface_normal(self, face):
        return self.normals[face]

    def fresnel_reflectance(self, faces, i):
        """
        Returns Fresnel reflectance for each face

        face : face indices
        i : incident angle

        """
        n1 = self.n
        n2 = zeros(shape(faces))
        for uniqueface in set(faces):
            n2[faces == uniqueface] = self.mat[uniqueface].n

        r = arcsin(n1 / n2 * sin(i))

        r[isnan(r)] = pi / 2

        rTE = (n1 * cos(i) - n2 * cos(r)) / (n1 * cos(i) + n2 * cos(r))
        rTM = (n2 * cos(i) - n1 * cos(r)) / (n2 * cos(i) + n1 * cos(r))

        rTE[isnan(rTE)] = 1
        rTM[isnan(rTM)] = 1
        rTE[n1 == n2] = 0
        rTM[n1 == n2] = 0

        return 0.5 * (rTE ** 2 + rTM ** 2)

    def plot_frame(self, axis, threshold=1e-15, offset=(0, 0, 0), verbose=0):
        """
        plots a 3D wireframe of the box defined by the vertices and surface normals given
        Threshold : instead of ==0 we use < Threshold to allow for inaccuracies
        OffSet : shift in (x,y,z) to translate the frame where we want it
        """

        # Potential set of facets defining facet
        for X in combinations(self.corners, 4):
            if verbose > 0:
                print(X)
            Lines = [subtract(a, b) for a, b in combinations(X, 2)]
            OnFacet = [[abs(dot(line, s)) <= threshold for line in Lines]
                       for s in self.normals]

            OnFacet = [all(row) for row in OnFacet]

            if any(OnFacet):
                Pairs = [(a, b) for a, b in combinations(X, 2)]
                Lengths = [linalg.norm(line) for line in Lines]

                Pairs = zip(Lengths, Pairs)
                Pairs = sorted(Pairs)
                N = 4

                for L, p in Pairs[:N]:
                    q = [subtract(val, offset) for val in p]
                    x, y, z = zip(*q)
                    axis.plot(x, y, z, 'k-', lw=2, alpha=0.2)

        axis.set_title(self)  # overly complicated nonsense you pillock
        labelaxes(axis)


def two_infinite_faces(L, rindex, materials, **kwargs):
    """
    Two Infinite Faces separated by a distance L
    """
    Normals = [[1, 0, 0], [-1, 0, 0]]
    Points = [[L, 0, 0], [0, 0, 0]]
    Corners = [[0, 0, 0], [L, 0, 0], [0, L, 0],
               [L, L, 0], [0, 0, L], [L, 0, L], [0, L, L], [L, L, L]]
    Facenames = {0: "positive x", 1: "negative x"}
    return (
        Box(rindex, Corners, Normals, Points, Facenames,
            materials, name="Two Infinite Faces", **kwargs)
    )


def regular_cuboid(LX, LY, LZ, rindex, materials, **kwargs):
    """
    Creates regular cuboid
    LX,LY,LZ : lengths of x,y,z
    """
    Normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0],
               [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    Points = [[LX, LY, LZ], [0, 0, 0], [LX, LY, LZ],
              [0, 0, 0], [LX, LY, LZ], [0, 0, 0]]
    Corners = [[0, 0, 0], [LX, 0, 0], [0, LY, 0], [0, 0, LZ],
               [LX, LY, 0], [LX, 0, LZ], [0, LY, LZ], [LX, LY, LZ]]
    Facenames = {
        0: "positive x",
        1: "negative x",
        2: "positive y",
        3: "negative y",
        4: "positive z",
        5: "negative z"}
    return (
        Box(rindex, Corners, Normals, Points, Facenames,
            materials, name="Regular Cuboid", **kwargs)
    )


def raised_top_cuboid(LX, LY, LZ, ThetaX, rindex, materials, **kwargs):
    """
    Creates box with raised top edge
    LX,LY,LZ : lengths of x,y,z
    ThetaX : Angle box deviates from normal
    """
    LZPrime = LZ + LX * tan(ThetaX)
    Normals = [[1, 0, 0], [-1, 0, 0], [0, 1, 0],
               [0, -1, 0], [-sin(ThetaX), 0, cos(ThetaX)], [0, 0, -1]]
    Points = [[LX, LY, LZ], [0, 0, 0], [LX, LY, LZ],
              [0, 0, 0], [LX, LY, LZ], [0, 0, 0]]
    Corners = [[0, 0, 0], [LX, 0, 0], [0, LY, 0], [0, 0, LZ],
               [LX, LY, 0], [LX, 0, LZPrime], [0, LY, LZ], [LX, LY, LZPrime]]
    Facenames = {
        0: "positive x",
        1: "negative x",
        2: "positive y",
        3: "negative y",
        4: "positive z",
        5: "negative z"}
    return (
        Box(rindex, Corners, Normals, Points, Facenames,
            materials, name="Raised Top Edge", **kwargs)
    )


def trapezium_cuboid(LX, LY, LZ, ThetaX, ThetaY, rindex, materials, **kwargs):
    """
    Creates irregular trapezium
    LX,LY,LZ : lengths of x,y,z
    ThetaX,ThetaY : Angles defining deviation from regular cuboid
    """
    DX = LZ * tan(ThetaX)
    DY = LZ * tan(ThetaY)
    Corners = [[0, 0, 0], [LX, 0, 0], [0, LY, 0], [LX, LY, 0],
               [-DX, -DY, LZ], [LX + DX, -DY, LZ], [-DX, LY + DY, LZ], [LX + DX, LY + DY, LZ]]
    Normals = [[-cos(ThetaX), 0, -sin(ThetaX)], [cos(ThetaX), 0, -sin(ThetaX)],
               [0, -cos(ThetaY), -sin(ThetaY)], [0, cos(ThetaY), -sin(ThetaY)], [0, 0, 1], [0, 0, -1]]
    Points = [[LX, LY, LZ], [0, 0, 0], [LX, LY, LZ],
              [0, 0, 0], [LX, LY, LZ], [0, 0, 0]]
    Facenames = {
        0: "positive x",
        1: "negative x",
        2: "positive y",
        3: "negative y",
        4: "positive z",
        5: "negative z"}
    return (
        Box(rindex, Corners, Normals, Points, Facenames,
            materials, name="Trapezium", **kwargs)
    )
