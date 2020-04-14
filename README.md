# eht-dmc
A radio interferometric modeling tool developed for the Event Horizon Telescope (EHT).  The framework employed here utilizes PyMC3 for sampling and the eht-imaging library for VLBI data manipulation.

NOTE: Currently still in testing mode, so no guarantees that code will work at all yet.

### Installation instructions
To install the most recent version, clone the repository, move into the main directory, and install using `pip`:

```
git clone https://github.com/dpesce/eht-dmc.git
cd eht-dmc

sudo apt-get -y install libnfft3-dev  # for pynfft
#sudo apt-get -y install libopenblas-base libopenblas-dev  # need some blas for numpy

pip install cython  # might be needed by pynfft, must be installed before requirements.txt
pip install -r requirements.txt
pip install .
```

### Recommendations
Import the `eht-dmc` library using the `dm` alias:

```
import eht_dmc as dm
```

### Additional installation clues

For yum-based installs, the EPEL package nfft-devel does not contain
the threaded library that pynfft wants to use.

For pyenv-installed pythons, the Theano package requires that python be
installed with shared libraries, which is not the pyenv default.
Reinstall python like this:

```
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --force 3.X.Y
```

