#!/usr/bin/env python3

import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]


setup(
    name="unboxapi",
    version="0.0.5",
    description="The official Python client library for Unbox AI, the Testing and Debugging Platform for AI",
    url="https://github.com/unboxai/unboxapi-python-client",
    author="Unbox AI",
    license="BSD",
    packages=find_packages(exclude=["js", "node_modules", "tests"]),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
