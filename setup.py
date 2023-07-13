from setuptools import setup, find_packages

setup(
    name="pso2keras",
    version="0.1.2",
    description="Particle Swarm Optimization to tensorflow package",
    author="pieroot",
    author_email="jgbong0306@gmail.com",
    url="https://github.com/jung-geun/PSO",
    install_requires=[
        "tqdm==4.65.0",
        "tensorflow==2.11.1",
        "keras==2.11.0",
        "numpy",
        "pandas",
        "ipython",
    ],
    packages=find_packages(exclude=[]),
    keywords=["pso", "tensorflow", "keras"],
    python_requires="==3.8",
    package_data={},
    zip_safe=False,
    long_description=open("README.md", encoding="UTF8").read(),
    long_description_content_type="text/markdown",
)
