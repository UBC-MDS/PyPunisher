import os
from setuptools import setup, find_packages


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return 'See https://github.com/UBC-MDS/Punisher/tree/master'


setup(
    name='pypunisher',
    version='0.0.1',
    author='...',
    author_email='...',
    description='',
    long_description=read('docs/README.md'),
    license='BSD-3',
    keywords='...',
    url='...',
    download_url='...',
    packages=find_packages(),
    install_requires=[],
    classifiers=['Development Status :: 3 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)
