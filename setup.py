import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(name="cqmod",
          version = "0.1dev",
          
          author = "Dom Pesce",
          author_email = "dpesce@cfa.harvard.edu",
          description = ("Python code to perform radio interferometric imaging and "+
                         "modeling within a Bayesian framework in both "+
                         "total intensity and polarization."),
          keywords = "interferometry",
          url = "https://github.com/dpesce/DMC",
          packages = ["DMC"],
          long_description=read('README.txt'),
          install_requires=["ehtim",
                            "pymc3",
                            "future",
                            "matplotlib",
                            "multiprocessing",
                            "numpy",
                            "scipy",
                            "corner"]
    )
