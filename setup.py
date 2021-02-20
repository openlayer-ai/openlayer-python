#!/usr/bin/env python3

import pathlib
import pkg_resources
from setuptools import setup, find_packages

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(name='bentowrapper',
      version='0.0.1',
      description='Wrapper for easy BentoML usage',
      url='https://github.com/unboxai/bento-wrapper',
      author='Unbox AI',
      license='BSD',
      packages=find_packages(exclude=['js', 'node_modules', 'tests']),
      python_requires='>=3.5',
      install_requires=install_requires,
      include_package_data=True,
      zip_safe=False)
