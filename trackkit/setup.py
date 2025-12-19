from setuptools import setup, find_packages

setup(
    name="trackkit",               
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "pydantic>=2.0",
    ],
    include_package_data=True,
    description="Utilities for track feature preprocessing, summarization, and plotting",
    author="Emanuele Coradin",
    author_email="emanuele.coradin01@gmail.com",
)
