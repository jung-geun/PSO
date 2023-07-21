from setuptools import setup, find_packages

import pso

VERSION = pso.__version__

setup(
    name="pso2keras",
    version=VERSION,
    description="Particle Swarm Optimization to tensorflow package",
    author="pieroot",
    author_email="jgbong0306@gmail.com",
    url="https://github.com/jung-geun/PSO",
    install_requires=[
        "tqdm",
        "numpy",
        "pandas",
        "ipython",
    ],
    packages=find_packages(exclude=[]),
    keywords=["pso", "tensorflow", "keras"],
    python_requires=">=3.8",
    package_data={},
    zip_safe=False,
    long_description=open("README.md", encoding="UTF8").read(),
    long_description_content_type="text/markdown",
)
