from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='2dto3dto2d',
    version='0.1',
    packages=find_packages(),
    install_requires=required,
)