import setuptools

import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []

if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLTA",
    version="0.0.3",
    author="DeeperTrade",
    author_email="chakrit.y@deepertrade.com",
    description="Technical Analysis enhanced through Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/palmbook/MLTA",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires
)