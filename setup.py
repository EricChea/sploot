"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import codecs
from os import path

from setuptools import setup, find_packages

CURR_DIR = path.abspath(path.dirname(__file__))
with codecs.open(path.join(CURR_DIR, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='sploot',
    version='X.YaN',
    description='Machine Learning utilities',
    long_description=LONG_DESCRIPTION,
    author='DoDoSmarts',
    author_email='dodosmarts@protonmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers :: Data Folks',
        'Topic :: Machine Learning :: Recommender :: Forecasting :: Inventory',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ML',
    packages=find_packages(include=['sploot', 'sploot.*']),
    install_requires=['numpy', 'sklearn', 'scipy', 'h5py'],
)
