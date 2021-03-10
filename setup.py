import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="eht-dmc",
          version = "0.2",
          author = "Dom Pesce",
          author_email = "dpesce@cfa.harvard.edu",
          description = ("Python code to perform radio interferometric imaging and "+
                         "modeling within a Bayesian framework in both "+
                         "total intensity and polarization."),
          license = "GPLv3",
          keywords = "radio interferometry VLBI",
          url = "https://github.com/dpesce/eht-dmc",
          packages = ["eht_dmc"],
          long_description=read('README.md'),
          install_requires=["ehtim==1.2.2",  # https://github.com/achael/eht-imaging.git
                            "pymc3",
                            "arviz==0.10",
                            "future",
                            "matplotlib",
                            "numpy",
                            "matplotlib",
                            "scipy",
                            "corner",
                            "tqdm"]
    )
