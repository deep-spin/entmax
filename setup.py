from distutils.core import setup


setup(name='entmax',
      url="https://github.com/deep-spin/entmax",
      author="Ben Peters, Goncalo M Correia, Vlad Niculae",
      author_email="vlad@vene.ro",
      description=("The entmax mapping and its loss, a family of sparse "
                   "alternatives to softmax."),
      license="MIT",
      packages=['entmax'],
      install_requires=['torch>=1.0'],
      python_requires=">=3.5")
