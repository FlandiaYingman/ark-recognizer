from pathlib import Path

import setuptools

setuptools.setup(
    name="ark-recognizer-py",
    version="0.0.0",
    author="Flandia Yingman",
    author_email="Flandia_YingM@hotmail.com",
    description="Recognizes the type and number of items from Arknights screenshots.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/FlandiaYingman/ark-recognizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
)
