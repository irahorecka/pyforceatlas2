"""
setup
~~~~~
"""

import re
from setuptools import setup, find_packages


def find_version():
    """Function to find the version string in the __init__.py file."""
    with open("risk/__init__.py", "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="pyforceatlas2",
    version=find_version(),  # Dynamically fetches the version from the __init__.py file.
    author="Ira Horecka",
    author_email="ira89@icloud.com",
    description="Fast Gephi's ForceAtlas2 graph layout algorithm written in Python for Python.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
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
