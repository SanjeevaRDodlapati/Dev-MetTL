from glob import glob
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='transfer',
      version='1.0.0',
      description='Transfer learning for predicting CpG methylation',
      long_description=read('README.rst'),
      author='Sanjeeva Reddy Dodlapati',
      author_email='sdodl001@odu.edu',
      license="MIT",
      url='https://github.com/ODU-CSM/Pub-Met-TL',
      packages=find_packages(),
      scripts=glob('./scripts/*.py'),
      install_requires=['h5py',
                        'argparse',
                        'scikit-learn',
                        'scipy==1.7.3',
                        'pandas',
                        'numpy',
                        'pytest',
                        'tensorflow',
                        'matplotlib',
                        'seaborn'],
      keywords=['Transfer learning',
                'Deep neural networks',
                'Epigenetics',
                'DNA methylation',
                'Single cells'],
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   ]
      )
