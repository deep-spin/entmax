from setuptools import setup
import os
import sys

proj_path = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(proj_path, "entmax", "version.py")) as f:
    exec(f.read(), version)

setup(name='entmax',
      version=version["__version__"],
      url="https://github.com/deep-spin/entmax",
      author="Ben Peters, Goncalo M Correia, Vlad Niculae",
      author_email="vlad@vene.ro",
      description=("The entmax mapping and its loss, a family of sparse "
                   "alternatives to softmax."),
      license="MIT",
      packages=['entmax'],
      install_requires=['torch>=1.0'],
      python_requires=">=3.5")
