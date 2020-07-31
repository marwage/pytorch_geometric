import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="benchmarking",
    version="0.0.1",
    author="Marcel Wagenländer",
    author_email="marcel.wagenlaender@tum.de",
    description="Benchmarking GNNs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://www.marcelwagenlaender.net/benchmarking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
