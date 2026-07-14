from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Omnibrain",
    packages=find_packages(exclude=['notebooks', 'logs', 'artifacts', 'tests*']), 
    install_requires = requirements, 
)