from __future__ import print_function, division

from numpy import cumsum

from .const import *


class Surface():
    """
    Surface material description

    n : refractive index (for critical angle and fresnel)
    ref : absorptivity parameter
    name : name of material

    --kwargs--
    surface :  parameters defining probability of each kind of reflection
    angle insensitive, overridden by specular_only keyword in main function
    ('Specular', 'Lobe', 'Backscatter', 'Lambertian', 'Segment')

    lobeangle : standard deviation of Gaussian Lobe Reflection
    mintheta, maxtheta : spherical segment range of angles
    """

    def __init__(self, n, ref=0, name="", **kwargs):
        self.n = n  # refractive index of material
        self.name = name  # name of material
        self.reflectivity = ref

        ## Enable these interactions during face_escape_status?
        self.critical = kwargs.get('critical', True)
        self.fresnel = kwargs.get('fresnel', True)
        self.ref = self.reflectivity > 0

        self.lobeangle = kwargs.get('lobeangle', 1.3 * Degrees)  # doi: 10.1109/TNS.2010.2042731

        surface_parameters = kwargs.get('surface', (1, 0, 0, 0, 0))
        surface_parameters = cumsum(surface_parameters) / sum(surface_parameters)
        self.surface = surface_parameters  # defaults to specular

        self.mintheta = kwargs.get('mintheta', 0 * Degrees)
        self.maxtheta = kwargs.get('maxtheta', 90 * Degrees)

        self.minphi = kwargs.get('minphi', None)
        self.maxphi = kwargs.get('maxphi', None)

        assert self.mintheta >= 0, "Minimum theta is negative - not allowed!"
        assert self.maxtheta <= 90 * Degrees, "Maximum theta is > 90 - not allowed!"

    def __repr__(self):
        return self.name


#defaults to 6 sided box
def OneMaterial(materialA, faces=range(6)):
    return {fa: materialA for fa in faces}


def TwoMaterials(materialA, materialB, facesA=([5]), facesB=range(5)):
    A = {fa: materialA for fa in facesA}
    B = {fb: materialB for fb in facesB}

    return dict(A.items() + B.items())


def ThreeMaterials(materialA, materialB, materialC, facesA=([5]), facesB=([3]), facesC=(0, 1, 2, 4)):
    A = {fa: materialA for fa in facesA}
    B = {fb: materialB for fb in facesB}
    C = {fb: materialC for fb in facesC}

    return dict(A.items() + B.items() + C.items())



