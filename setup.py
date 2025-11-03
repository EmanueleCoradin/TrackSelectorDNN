from setuptools import setup, find_packages

setup(
    name="TrackSelectorDNN",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pydantic>=2.0",
    ],
    include_package_data=True,
    package_data={
        "TrackSelectorDNN.configs": ["*.yaml"],
    },
)
