from distutils.core import setup

_VERSION = "0.4.0"

with open("README", "r") as readme:
    long_desc = readme.read()


setup(
    version=_VERSION,
    name="time-series-lasso",
    packages=["ts_lasso", "test"],
    author="Ryan J. Kinnear",
    author_email="Ryan@Kinnear.ca",
    description=("FISTA implementation specialized for VAR(p) models."),
    long_description=long_desc,
    url="https://github.com/RJTK/time-series-lasso",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"],
    license="LICENSE"
)
