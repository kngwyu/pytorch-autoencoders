from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

requirements = [
    'numpy>=1.15.0',
    'matplotlib>=3.0',
    'opencv-python>=3.4',
    'torch>=1.0',
    'torchvision>=0.2.1',
]
test_requirements = ['pytest>=3.0']

setup(name='pytorch_autoencoders',
      author='Yuji Kanagawa',
      version='0.1',
      packages=find_packages(),
      install_requires=requirements,
      test_requires=test_requirements)
