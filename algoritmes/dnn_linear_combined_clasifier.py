__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"

from dnn import Dnn
from common import *

import tensorflow as tf

class DnnLinearCombinedClassifier(Dnn):
    """
    implements a deep and wide neural network
    see https://medium.com/@camrongodbout/tensorflow-in-a-nutshell-part-two-hybrid-learning-98c121d35392#.h8nx7xjc9
    for more info
    """

    def build(self, definition):
        """
        build the model based on the definition
        :param definition: a json structure that declares the model
        :return: None
        """
        for feature in definition['features']:
            if 'asset' in feature:
                """create feature for asset"""
                self.buildAssetFeature(feature, getFeatureName(feature, feature['asset']))
            elif 'time' in feature:
                """create a feature based on event timing"""
                self.buildTimeFeature(feature, getFeatureName(feature, feature['time']))
            elif 'crossed' in feature:
                """create a feature for a crossed column (combine multiple columns into 1"""
                self.buildCrossedFeature(feature, getFeatureName(feature, None))
            else:
                raise Exception("unknown feature type, currently supported: asset, time, crossed")
        self.buildResult(definition)

        nrClasses = self.getNrClasses(definition['result'])
        self.model = tf.contrib.learn.DNNClassifier(feature_columns=self.features.values(),
                                                    # we need a list of the columns, not a dict of names and columsn
                                                    hidden_units=definition['hidden_units'],
                                                    n_classes=nrClasses,
                                                    model_dir=self.model_dir)
        return self.model