from setuptools import setup
import glob


setup_args = dict( name="dl-utils",
    version="0.0.1",
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

    packages=[ "dl_utils" ],

    include_package_data = False,       # otherwise package_data is not used
)

if __name__ == '__main__':
    setup( **setup_args )
