from setuptools import find_packages, setup

setup(
    name="serval-decode",
    version="0.1",
    description="Toolboox for decoding multiplexed spatial transcriptomics imaging data.",
    author="Andrew Roth",
    author_email="aroth@bccr.ca",
    url="https://github.com/Roth-Lab/serval",
    packages=find_packages(),
    license="GPL v3",
    install_requires=["numpy", "pandas", "scikit-image", "scipy"],
)
