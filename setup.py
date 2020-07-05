from distutils.util import convert_path
from skbuild import setup

main_ns = {}
ver_path = convert_path('pynegf/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pynegf",
    version=main_ns['__version__'],
    author="Gabriele Penazzi",
    author_email="g.penazzi@gmail.com",
    description="A wrapper for libnegf, a library for Non Equilibrium Green's Function based charge transport.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpenazzi/pynegf",
    packages=['pynegf'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy'],
    tests_require=['pytest'],
    cmake_args=[
        '-DBUILD_SHARED_LIBS=ON'],
    cmake_source_dir='libnegf',
    cmake_install_dir='pynegf'
)
