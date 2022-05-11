import glob
from setuptools import setup

# scripts_path = 'scripts/*.py'
# file_list = glob.glob(scripts_path)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='convect=py',
      version='0.1',
      description='Python code that solves the 2D non linear convection',
      long_description=long_description,
      long_description_content_type="text/markdown",
#      url='https://github.com/jnywong/nondim-slurry',
#      license='LICENSE.md',
      author='Jenny Wong',
      author_email='j.wong.1@bham.ac.uk',
      packages=['src'],
      # install_requires=[<pypi_package_name>],
      # scripts=file_list,
#      package_data={'slurpy': ['lookupdata/*.csv', 'scripts/*.py']},
    #   install_requires=['pytest'],
      )