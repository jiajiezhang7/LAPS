import os
import pathlib
from setuptools import setup, find_packages

root_dir = os.path.dirname(os.path.realpath(__file__))

HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = ["torch", "omegaconf", "hydra-core"]

setup(
    name='amplify',
    version='1.0.0',
    author='Jeremy Collins and Loránd Cheng',
    author_email='jer@gatech.edu, lorand@gatech.edu',
    description='Actionless Motion Priors for Robot Learning from Videos',
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=["robotics", "robot learning", "world models", "point track prediction"],
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    packages=find_packages(),
    # package_dir={"": "amplify"},
    zip_safe=False,
)