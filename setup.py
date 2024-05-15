from setuptools import find_packages, setup

import pso

VERSION = pso.__version__


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="pso2keras",
    version=VERSION,
    description="Particle Swarm Optimization on tensorflow package",
    author="pieroot",
    author_email="jgbong0306@gmail.com",
    url="https://github.com/jung-geun/PSO",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(exclude=[]),
    keywords=[
        "pso",
        "tensorflow",
        "keras",
        "optimization",
        "particle swarm optimization",
        "pso2keras",
    ],
    python_requires=">=3.10",
    package_data={},
    zip_safe=False,
    long_description=open("README.md", encoding="UTF8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
