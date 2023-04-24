import os

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name="research",
    version="0.0.1",
    author="Philipp Wu",
    author_email="philippwu@berkeley.edu",
    description="mtm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/mtm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    license="MIT",
    install_requires=read_requirements_file("requirements.txt"),
)
