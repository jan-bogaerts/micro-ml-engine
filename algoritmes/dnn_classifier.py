__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"

import logging
logger = logging.getLogger('dnn_classifier')

import tensorflow as tf


import dnn
from common import *


class DnnClassifier(dnn.Dnn):
    """
    wraps a dnnclassifier object
    """

    def __init__(self, connection, dir):
        """init the object
        :param connection: the client connection to use for this model (getting values)
        """
        super(DnnClassifier, self).__init__(connection, dir)


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
                raise Exception("crossed columns are not supported in a DNN classifier, use a linear combined classifier (deep and wide NN) instead.")
            elif 'source' in feature:
                """create a feature that is a derivative of another column. This is used for a feature that is bucketized, but who's original value is also used."""
                self.buildFeatureFromSource(feature, getFeatureName(feature, "from_source_" + feature['source']))
            else:
                raise Exception("unknown feature type, currently supported: asset, time, crossed")
        self.buildResult(definition)

        nrClasses = self.getNrClasses(definition['result'])
        self.model = tf.contrib.learn.DNNClassifier(feature_columns=self.features.values(),     # we need a list of the columns, not a dict of names and columsn
                                                         hidden_units=definition['hidden_units'],
                                                         n_classes=nrClasses,
                                                         model_dir=self.model_dir)
        return self.model

    def getNrClasses(self, result):
        """
        extracts the nr of classes fromm the asset that will store the result. This is based on the data type and enum/labels (profile)
        :param result: a json structure defining the result asset (currently only a direct asset id is supported.
        :return: an integer, representing the nr of items that the model can discover
        """
        if 'actuator' in result:
            name = result['actuator']
        elif 'sensor' in result:
            name = result['sensor']
        elif 'asset' in result:
            name = result['asset']
        else:
            raise Exception("unknown result sensor, can't build model")
        device = result['device']
        assetDef = self.connection.getAsset(name=name, device=device)
        profile = assetDef['profile']
        dataType = profile['type']
        if dataType == 'boolean':
            return 2
        elif 'enum' in profile:
            return len(profile['enum'])
        else:
            control = assetDef['control']
            if 'extras' in control:
                extras = control['extras']
                if 'labels' in extras:
                    return len(extras['labels'])
        raise Exception("can't extract the nr of supported results, either use a boolean as result asset, or declare an enum or set of labels")
