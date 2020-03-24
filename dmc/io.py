#######################################################
# imports
#######################################################

from __future__ import division
from __future__ import print_function
from builtins import list
from builtins import len
from builtins import range
from builtins import enumerate
from builtins import Exception

import numpy as np
import ehtim as eh
import pickle

#######################################################
# functions
#######################################################

def save_model(modelinfo,outfile):
    """ Save a model as a binary pickle file

       Args:
           modelinfo (modelinfo): dmc modelinfo object
           outfile (str): name of output file
           
       Returns:
           None
    
    """

    # saving the model file
    pickle.dump(modelinfo,open(outfile,'wb'),protocol=pickle.HIGHEST_PROTOCOL)

def load_model(infile):
    """ Load a model saved with save_model

       Args:
           infile (str): name of model file
           
       Returns:
           modelinfo: dmc modelinfo object
    
    """

    # loading the model file
    modelinfo = pickle.load(open(infile,'rb'))

    return modelinfo


