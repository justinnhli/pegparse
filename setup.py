from setuptools import setup

description = ''
try:
    with open('README.md', 'r') as f:
        description = f.read()
except FileNotFoundError:
    pass

setup(
    name='pegparse',
    version='0.0.1',
    description='A Parsing Expression Grammar Parser',
    long_description=description,
    author='Justin Li',
    author_email='justinnhli@gmail.com',
    license='MIT',
    packages=['pegparse'],
    url='https://github.com/justinnhli/pegparse',
    install_requires=[],
    entry_points={},
)
