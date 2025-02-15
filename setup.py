"""
setup
~~~~~
"""

import os
from setuptools import setup, find_packages


# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyforceatlas2",
    version="0.0.1",
    author="Ira Horecka",
    author_email="ira89@icloud.com",
    description="Fast Gephi's ForceAtlas2 graph layout algorithm written in Python for Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/irahorecka/pyforceatlas2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx",
        "numpy",
        "python-igraph",
        "scipy",
        "tqdm",
    ],
)
