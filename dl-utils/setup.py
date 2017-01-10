from setuptools import setup
import glob
import os.path

def pkg_version():
    basedir = os.path.dirname( os.path.realpath(__file__) )
    with open( os.path.join(basedir,'VERSION.txt'), 'r' ) as f:
        return f.readline().strip()


setup_args = dict( name="dl-utils",
    version=pkg_version(),
    author="Paulo Villegas",
    author_email="paulo.vllgs@gmail.com",
    
    description="Miscellaneous tiny utils to help working with ML/DL tasks in an IPython Notebook context",
    url="https://github.com/paulovn/docker-dl-gpu",
    platforms=["any"],
    classifiers=["Development Status :: 4 - Beta",
                 "Environment :: Console",
                 "Intended Audience :: Developers",
                 "Programming Language :: Python",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: BSD License",
                 "Operating System :: OS Independent",
                 ],
 
    install_requires = [ 'setuptools',
    ],

    packages=[ "dl_utils", "dl_utils.krs" ],

    include_package_data = False,       # otherwise package_data is not used
)

if __name__ == '__main__':
    setup( **setup_args )
