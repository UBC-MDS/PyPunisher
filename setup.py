from os.path import join, dirname
from setuptools import setup, find_packages


def read(fname):
    try:
        return open(join(dirname(__file__), fname)).read()
    except:
        return 'See https://github.com/UBC-MDS/PyPunisher/tree/master'


setup(
    name='pypunisher',
    version='4.0.1',
    author='Jill Cates, Avinash Prabhakaran, Tariq Hassan',
    author_email='NA',
    description='Model Selection in Python',
    long_description=read('docs/README.md'),
    license='BSD-3',
    keywords='model selection',
    url='https://github.com/UBC-MDS/PyPunisher',
    download_url='https://github.com/UBC-MDS/PyPunisher/archive/v4.0.1tar.gz',
    packages=find_packages(exclude=("tests",)),
    # Note: requirements.txt contains some packages
    # which are not needed to simply use the package
    # (i.e., they're only need to execute tests, e.g., `pytest`).
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Natural Language :: English',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3.6',
                 'License :: OSI Approved :: BSD License'
    ],
    include_package_data=True
)
