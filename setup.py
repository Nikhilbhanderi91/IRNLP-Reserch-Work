"""
================================================
setup.py
Package setup for HMRP project
================================================
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = [
        line.strip() for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="hmrp",
    version="1.0.0",
    author="HMRP Project",
    description=(
        "Hierarchical Concept Mapping for Research Paper Similarity Analysis"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["data", "logs", "tests*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "hmrp=run:launch_streamlit",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
