from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='basicnnet',
   version='1.0',
   description='A basic neural network',
   license="GNU General Public License v3.0",
   long_description=long_description,
   author='Chris Verdence',
   author_email='chris.verdence@gmail.com',
   url="https://github.com/cverdence",
   packages=['basicnnet'],
)