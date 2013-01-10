#!/home/amitibo/epd/bin/python

"""
atmotomo: A toolbox for atmospheric tomography.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://bitbucket.org/amitibo/atmosphere>
License: See attached license file
"""

from setuptools import setup
import os

NAME = 'atmotomo'
PACKAGE_NAME = 'atmotomo'
PACKAGES = [PACKAGE_NAME]
VERSION = '0.1'
DESCRIPTION = 'A toolbox for atmospheric tomography.'
LONG_DESCRIPTION = """
`atmotomo` is a toolbox for atmospheric tomography.
"""
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
KEYWORDS = []
LICENSE = 'GPLv3'
CLASSIFIERS = [
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Scientific/Engineering'
]
URL = "http://bitbucket.org/amitibo/atmosphere"

def main():
    """main setup function"""
    
    #
    # Create a source path file in the module base folder
    #
    with open(os.path.join(os.path.abspath(PACKAGE_NAME), '_src.py'), 'wb') as f:
        f.write('__src_path__=r"%s"\n' % os.getcwd())
    
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        license=LICENSE,
        packages=PACKAGES,
        package_data={PACKAGE_NAME: ['data/*.*']},
        scripts=[
            'scripts/simulateAtmo3D.py',
            'scripts/analyzeAtmo3D.py',
            'scripts/resultsAtmoGui.py'
        ]
        )


if __name__ == '__main__':
    main()
