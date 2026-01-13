"""
Setup script for the trackkit package.
"""

from setuptools import setup, find_packages

setup(
    # --------------------
    # Package metadata
    # --------------------
    name="trackkit",
    version="0.1",
    description="Utilities for track feature preprocessing, summarization, and plotting",
    author="Emanuele Coradin",
    author_email="emanuele.coradin01@gmail.com",

    # --------------------
    # Package contents
    # --------------------
    packages=find_packages(),
    include_package_data=True,

    # --------------------
    # Dependencies
    # --------------------
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "pydantic>=2.0",
        "scipy",
    ],
)
