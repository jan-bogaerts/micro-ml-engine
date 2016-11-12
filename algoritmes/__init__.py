__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"

from dnn_classifier import DnnClassifier
from dnn_linear_combined_clasifier import DnnLinearCombinedClassifier
import os


def create(definition, connection, name):
    """
    creates a model object and returns it
    :param definition: the json definition defining the model
    :param connection:
    :return:
    """
    modelname = str(definition['model'])
    classConst = globals()[modelname]           # dynamically get the class object so we can create an instance of the object
    res = classConst(connection, os.path.join('models', name))
    res.build(definition)
    return res