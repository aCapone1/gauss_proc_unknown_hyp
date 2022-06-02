from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))
#
# with open(path.join(current_dir, 'README.rst'), 'r') as f:
#     long_description = f.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')

setup(
    license='MIT',
    setup_requires='numpy',
    install_requires=install_requires
)
