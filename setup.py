#!/home/amitibo/epd/bin/python

"""
atmotomo: A toolbox for atmospheric tomography.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://bitbucket.org/amitibo/atmosphere>
License: See attached license file
"""

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import platform
import shutil
import stat
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

SHDOM_SRC = 'src/shdom'


def make_shdom(shdom_dir, script_dir):
    
    if platform.system() == 'Windows':
        raise Warning('Installing on Windows, SHDOM not installed.')
    
    current_path = os.getcwd()
    os.chdir('src/shdom')
    os.system('make')
    os.chdir(current_path)

    for name in ('shdom', 'shdom90', 'propgen', 'make_mie_table', 'gradient', 'extinctgen'):
        src_path = os.path.join(shdom_dir, name)
        dst_path = os.path.join(script_dir, name)
        shutil.copyfile(src_path, dst_path)
        os.chmod(dst_path, stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH | stat.S_IWUSR)
        
class SrcInstall(install):

    def run(self):
        install.run(self)
        make_shdom(SHDOM_SRC, self.script_dir)


class SrcDevelop(develop):

    def run(self):
        'running make'
        develop.run(self)
        make_shdom(SHDOM_SRC, self.script_dir)


def scripts_list():
    
    scripts = [
        'scripts/{name}.py'.format(name=name) for name in ('simulateAtmo3D', 'analyzeAtmo3D', 'simulateSHDOM', 'analyzeSHDOM', 'pbsSHDOM')
    ]

    return scripts


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
        scripts=scripts_list(),
        cmdclass={
            'install': SrcInstall,
            'develop': SrcDevelop
        }
    )


if __name__ == '__main__':
    main()
