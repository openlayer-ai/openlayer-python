#!/usr/bin/env python3

import pathlib
from distutils.util import convert_path

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

main_ns = {}
ver_path = convert_path("unboxapi/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="unboxapi",
    version=main_ns["__version__"],
    description="The official Python client library for Unbox, the Testing and Debugging Platform for AI",
    url="https://github.com/unboxai/unboxapi-python-client",
    author="Unbox",
    license="BSD",
    packages=find_packages(exclude=["js", "node_modules", "tests"]),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
