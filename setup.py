from distutils.core import setup
from entmax import __version__

setup(name='entmax',
      version=__version__,
      url="https://github.com/deep-spin/entmax",
      author="Ben Peters, Goncalo M Correia, Vlad Niculae",
      author_email="vlad@vene.ro",
      description=("The entmax mapping and its loss, a family of sparse "
                   "alternatives to softmax."),
      license="MIT",
      packages=['entmax'],
      python_requires=">=3.5")
