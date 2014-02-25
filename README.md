Photon Tracing
==============

Description
-----------
Photon tracing is a Monte-Carlo simulation of photons within a closed geometry.
In this we endeavour to implement realistic surface models in an efficient manner.

The current implementation has :

* The Extended UNIFIED model for reflection 
(specular, backscatter, lobe, lambertian and spherical segment). This is currently
incident direction invariant.
* Arbitrary geometry model defined by surface normal and point. Several template
files included.
* Material class file to allow energy and polarisation properties to be included.


Roadmap
-------

1. Implement unit tests for known analytical solutions and physical behaviour.
A folder with ipython notebooks with each derivation that we implement in code.
2. Documentation and examples for simple cases.
3. Correct implementation of parallelised operation with profiling of CPU and
memory usage to ensure # of photons per process approaches optimal.
4. Allow photon property dependence to affect every interaction. Drude-Lorenz
model (and others!) model of material properties accurately. Correct implementation
of metals. Fate of photons should be recorded, thus allowing the definition of detectors
via absorption property.
5. Animation of energy density with time, along with power per unit time implemented
using Bokeh. Should be consistent with absorption property.
6. Vector geometries such that spherical shells and cylindrical shells can be implemented.
7. Abstract objects so we can define combined objects.




