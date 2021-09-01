from setuptools import find_packages
from distutils.core import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

# cpp_arr = os.listdir('fle/cpp')
# s = ['fle/cpp/' + a for a in cpp_arr if a[-3:]=='cpp']
# eigen_arr = os.listdir('fle/cpp/Eigen')
# s += ['fle/cpp/Eigen/' + a for a in eigen_arr if len(a) > 1 ]

#s = getListOfFiles('fle/cpp')
#s.remove('fle/cpp/.gitignore')
#print(s)
module1 = Pybind11Extension('flocking_cpp',
                    #sources = s)
                    sources = ['fle/cpp/py_interface.cpp'],
                    extra_compile_args = ["-march=native"])

setup(
    name='fle',
    version=0.1,
    author='Swarm Labs',
    author_email="justinkterry@gmail.com",
    description="Flocking Learning Environment",
    url="http://github.com/jkterry1/birdflocking",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "gym"],
    options={"bdist_wheel": {"universal": True}},
    #python_requires=">=3.6, <3.10",
    packages=["fle"] + ["fle." + pkg for pkg in find_packages("fle")],
    include_package_data=True,
    install_requires=[
        "pettingzoo",
        "pybind11"
    ],
    setup_requires=['pybind11>=2.2'],
    # classifiers=[
    #     'Programming Language :: Python :: 3'],
    cmdclass={"build_ext": build_ext},
    ext_modules = [module1]
)
