#!/bin/bash
set -e -x

# CLI arguments
PY_VERSIONS="cp38-cp38" #"cp37-cp37m" "cp36-cp36m"
BUILD_REQUIREMENTS=gfortran
SYSTEM_PACKAGES=
BUILD_REQUIREMENTS="scikit-build"
#PACKAGE_PATH=$4
PIP_WHEEL_ARGS=$5

if [ ! -z "$SYSTEM_PACKAGES" ]; then
    yum install -y ${SYSTEM_PACKAGES}  || { echo "Installing yum package(s) failed."; exit 1; }
fi

# Checkout package
git clone https://github.com/gpenazzi/pynegf.git && cd pynegf
git submodule update --init --recursive

# Compile wheels
arrPY_VERSIONS=(${PY_VERSIONS// / })
for PY_VER in "${arrPY_VERSIONS[@]}"; do
    # Update pip
    /opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip

    # Check if requirements were passed
    if [ ! -z "$BUILD_REQUIREMENTS" ]; then
        /opt/python/"${PY_VER}"/bin/pip install --no-cache-dir ${BUILD_REQUIREMENTS} || { echo "Installing requirements failed."; exit 1; }
    fi

    # Build wheels
    cmake --version
    #/opt/python/"${PY_VER}"/bin/python setup.py install -- -G "Unix Makefiles" || { echo "Building wheels failed."; exit 1; }
    /opt/python/"${PY_VER}"/bin/python setup.py bdist_wheel -- -G "Unix Makefiles" || { echo "Building wheels failed."; exit 1; }
done

# Bundle external shared libraries into the wheels
for whl in /pynegf/dist/*-linux*.whl; do
    auditwheel repair "$whl" --plat "${PLAT}" -w dist || { echo "Repairing wheels failed."; auditwheel show "$whl"; exit 1; }
done

echo "Succesfully build wheels:"
ls dist
# Copying to local volume scratch
cp dist/*manylinux* /scratch/
