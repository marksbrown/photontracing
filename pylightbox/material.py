from __future__ import print_function, division
from numpy import cumsum
from .const import *


class Surface():
    '''
    Outer surface materials

    n : refractive index (for critical angle and fresnel)
    '''

    def __init__(self, n, ref, name="", **kwargs):
        self.n = n  # refractive index of material
        self.name = name  # name of material
        self.reflectivity = ref  # reflectivity 

        if 'unified' in kwargs:
            #UNIFIED Model : specular,lobe, back-scatter, Lambertian
            self.surface = cumsum(kwargs.get('unified', [1, 0, 0, 0, 0]))
            assert len(self.surface) == 5, "Incorrect number of parameters passed"
            self.lobeangle = kwargs.get('lobeangle', 0*Degrees)
        elif 'extended in kwargs':
            self.surface = cumsum(kwargs.get('extended', [1, 0, 0, 0, 0]))
            assert len(self.surface) == 5, "Incorrect number of parameters passed"
            self.lobeangle = kwargs.get('lobeangle', 0*Degrees)
            self.mintheta = kwargs.get('mintheta', 0*Degrees)
            self.maxtheta = kwargs.get('maxtheta', 90*Degrees)
            assert self.mintheta >= 0, "Minimum theta is negative - not allowed!"
            assert self.maxtheta <= 90*Degrees, "Maximum theta is > 90 - not allowed!"
        else:
            self.surface = [1, 0, 0, 0, 0]  # defaults to specular

    def __repr__(self):
        return self.name
        

#defaults to 6 sided box
def OneMaterial(materialA, faces=range(6)):
    return {fa : materialA for fa in facesA}

def TwoMaterials(materialA, materialB, facesA=([5]), facesB=range(5)):
    A = {fa : materialA for fa in facesA}
    B = {fb : materialB for fb in facesB}
    
    return dict(A.items()+B.items())

def ThreeMaterials(materialA, materialB, materialC, facesA=(5), facesB=(3), facesC=(0, 1, 2, 4)):
    A = {fa : materialA for fa in facesA}
    B = {fb : materialB for fb in facesB}
    C = {fb : materialC for fb in facesC}
    
    return dict(A.items()+B.items()+C.items())



