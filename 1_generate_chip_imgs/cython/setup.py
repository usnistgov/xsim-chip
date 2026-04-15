from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="img2stl",
        sources=["img2stl.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ),
]

setup(
    name="STL-Cython-Extension",
    ext_modules=cythonize(extensions, annotate=True),
)