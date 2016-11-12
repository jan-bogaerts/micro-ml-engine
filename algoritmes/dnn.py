__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"

import logging
logger = logging.getLogger('dnn')

import tensorflow as tf
import dateutil
import numpy as np

from att_event_engine.resources import Asset
from att_trusted_event_server.when_server import registerAssetToMonitor
from att_trusted_event_server.callbackObject import CallbackObject
from att_event_engine.resources import Sensor, Actuator


class AssetMapping(object):
    """
    keeps track of all the information for an asset feature: which columns were created, which filter to use, and optional delay
    """
    def __init__(self, colname, filter, value, is_sparse=False):
        self.column = colname
        self.filter = filter
        self.is_sparse = is_sparse
        self.value = value

class Dnn(object):
    """
    root class for all classes that use a neural network.
    """

    def __init__(self, connection, dir):
        """init the object
        :param connection: the client connection to use for this model (getting values)
        """
        self.model_dir = dir
        self.model = None
        self.connection = connection
        self.asset_mappings = {}  # stores a mapping between asset id's and the column names, filter, delay. Used when a new asset value arrives, so the system knows which columsn to populate and how to filter. Note: value is a single object, asset can only be used 1 time (unless bucketized, which can be used in multiple bucket versions)
        self.features = {}          # a dict containing all the features that were created for this model. Keys = column name. Note: this dict is also used to build crossed columsn
        self.time_mappings = {}  # stores a mapping between column name (value) and the time grouping required (time of day, day of week,...)
        self.result = None  # stores the actuator/sensor that will receive t
        self.training_steps = 1  # default nr of training steps used for when we receive an actuator command to trigger a training session.
        self.callback = CallbackObject(None, self.fit)

    def fit(self):
        """
        called when an asset has changed. Will send the values to the model and make a new prediction.
        :return: None
        """
        current = Asset.current()
        if current.id in self.asset_mappings:
            current.connection = self.connection  # need to make certain that we use the correct connection, otherwise it's the default, which is not connected.
            mapping = self.asset_mappings[current.id]
            value = current.value
            profile = current.profile
            dataType = profile['type']
            if dataType == 'boolean':
                mapping.value = int(value)  # we need to keep the original value for the filter functions
            else:
                mapping.value = value
            if not mapping.filter or eval(mapping.filter):
                result = self.model.predict(input_fn=lambda: self.build_input_data(current.value_at))
                if len(result) > 0:
                    if self.result.profile['type'] == "boolean":
                        output = bool(result[0])
                    else:
                        output = result[0]
                    self.result.updateState(output)  # the result comes as a list, but we only requested for 1 record, so the list only contains 1 item.

    def train_from_feedback(self, actuator, value):
        """
        called when the user has sent a new value to the result actuator in order to indicate that the last fit was
        incorrect and the new value should be used to train the system.
        :param actuator:
        :param value:
        :return:
        """
        try:
            profile = self.result.profile
            dataType = profile['type']
            if dataType == 'boolean':
                value = int(value)
            if value:
                self.model.fit(input_fn=lambda: self.build_input_data(actuator.value_at, value))
        except:
            logger.exception("failed to train model")

    def train(self, data, columns, steps):
        """
        train the model using the specified data.
        :param steps: the nr of steps to use for training.
        :param columns: a list with the column names, in the same order as the data provides.
        :param data: a list of data records. each record is a list itself.
        :return:
        """
        self.check_columns(columns)
        self.model.fit(input_fn=lambda: self.build_test_data(data, columns), steps=steps)

    def build_test_data(self, data, columns):
        """
        The test data tensors must be built within a callback function that is passed to the 'fit' function.
        otherwise, the system reports that the biasTensor is from another graph.
        :param data:
        :param columns:
        :return:
        """
        result = {}
        sparseFieldNames = self.get_sparse_field_names()
        data = np.array(data)
        colIndex = 0
        label = None
        for name in columns:
            colData = data[:, colIndex]
            if colData.dtype == np.bool:  # convert booleans to number
                colData = colData.astype(int)
            if colData.dtype == type(np.array):
                rowLen = len(colData[0])
            else:
                rowLen = 1
            isSparse = name in sparseFieldNames
            if name == 'result':
                label = tf.constant(colData, shape=[len(data), 1])
            else:
                self.store_in_feed_dict(result, name, colData, isSparse, len(data),
                                        rowLen)  # we also need to get the row length, in case that a single input value is also a vector
            colIndex += 1
        if label == None:  # when label is tensor, then 'not label' gives an error -> can't use tensor as bool
            raise Exception("no 'result' column defined, can't use data for training")
        return result, label

    def get_sparse_field_names(self):
        """
        builds up a set of the fieldnames that are designated to be sparse. This set is used during the training stage.
        The set needs to be built, cause the data is stored differently in the object (as a dict), which is more
         convinient for live data, but not for training from a rendered data-set.
        :return: a set
        """
        result = set()
        for key, value in self.asset_mappings.iteritems():
            if value.is_sparse:
                result.add(value.column)
        return result

    def check_columns(self, columns):
        """
        Check if the columns (provided during a training session) match all the required columns.
        :param columns: a list containing the column names that are supplied to the algoritme
        :return: None. Raises an exception if there is a column mismatch (too many, too few)
        """
        missing_cols = []
        too_many_cols = []
        new_cols = list(columns)
        if not 'result' in new_cols:
            missing_cols.append('result')
        else:
            new_cols.remove('result')
        features = set(self.time_mappings.values())  # we need the assets and time mappings, all the rest is derived from these. So don't use the features dict, cause it might have more then required fro the input due to derivied cols (cols that use another col as input)
        features.update([val.column for key, val in self.asset_mappings.iteritems() ])
        for col in new_cols:
            if not col in features:
                too_many_cols.append(col)
            else:
                features.remove(col)
        missing_cols.extend(features)
        errorMessage = ""
        if len(missing_cols) > 0:
            errorMessage = "The following columns are missing in the training data: {}".format(missing_cols)
        if len(too_many_cols) > 0:
            if len(errorMessage) > 0:
                errorMessage += "\n"
            errorMessage += "The following columns are not included in the model and should not be provided in the training data: {}".format(
                too_many_cols)
        if len(errorMessage) > 0:
            raise Exception(errorMessage)

    def buildResult(self, definition):
        """
        builds the actuator or sensor object that will receive the result value.
        In case of an actuator, the training mode can be activated.

        the result section should contain the following parts:
        - device: the id of the device that contains the asset that stores the result
        - sensor or actuator: the name of the asset to send the result to. When actuator is used, training can be done by sending an actuator command
        - steps: optional, only valid for actuators. the nr of steps that should be taken to train a new value. Default = 1

        :param definition: the definition of the model
        :return: None
        """
        if not 'result' in definition:
            raise Exception("no result decleration found in model definition")
        resultDef = definition['result']
        if 'steps' in resultDef:
            self.training_steps = resultDef['steps']
        if not 'device' in resultDef:
            raise Exception(
                "The result decleration is missind a 'device' field, which should contain the id of the device (only the form device-id/assetname is currently supported)")
        device = resultDef['device']
        if 'actuator' in resultDef:  # we support training
            self.result = Actuator(name=resultDef['actuator'], device=device, connection=self.connection)
            self.result.on_atuate = self.train_from_feedback
        elif 'sensor' in resultDef:
            self.result = Sensor(name=resultDef['sensor'], device=device, connection=self.connection)
        else:
            raise Exception(
                "the result decleration should contain an 'actuator' or 'sensor' field with the name of the asset.")

    def addFeatureColumnToDict(self, column, definition, name, is_sparse=False, is_bucketized=False):
        """
        adds the column to the list, taking into account if the user requested a random init.
        It also fills in the feed_dict with the specified value, if any.
        :param is_bucketized: When true, the feature is using buckets. When True, 'embedding vector dimension' is allowed
        :param column: the column to add
        :param definition: the definition for the feature
        :param name: the name of the feature column
        :return:
        """
        if 'embedding vector dimension' in definition:
            if is_sparse == False and is_bucketized == False:
                raise Exception("'embedding vector dimension' is only supported on sparse features, please define buckets or use an Enum asset type.")
            elif 'crossed' in definition:
                raise Exception("crossed columns don't support the field 'embedding vector dimension'.")
            dimension = definition['embedding vector dimension']
            self.features[name] = tf.contrib.layers.embedding_column(column, dimension=dimension)
        else:
            self.features[name] = column

    def store_in_feed_dict(self, result, col, value, is_sparse, colLength=1, rowlen=1):
        """
        store the specified value in the feed_dict so that it can be used to train the network or evaluate the values in the feed dict.
        :param result: the dictionary that stores the result.
        :param col: the name of the column to update
        :param value:  the value to store for the column
        :param colLength: the nr of elements in the column
        :param is_sparse: true if the value should be stored as a sparse value (for labels and such)
        :return: None
        """
        if colLength > 1:
            if not is_sparse:
                result[col] = tf.constant(value, shape=[colLength, rowlen])
            else:
                result[col] = tf.SparseTensor(indices=[0, 0], values=[value], shape=[colLength, rowlen])
        else:
            if not is_sparse:
                result[col] = tf.constant([value], shape=[colLength, rowlen])
            else:
                result[col] = tf.SparseTensor(indices=[0, 0], values=[value], shape=[colLength, rowlen])

    def fill_timestamp_in_feed_dict(self, result, timestamp):
        """
        convert the timestamp to the required time formats and fill into the feed_dict so that they can be used for a prediction.
        :param result: the dictionary that will store the result.
        :param timestamp: the timestamp as a string.
        :return: None
        """
        t = dateutil.parser.parse(timestamp)
        for time_format, name in self.time_mappings.iteritems():
            if time_format == "time_of_day":
                value = (t.hour * 3600) + (t.minute * 60) + t.second
            elif time_format == "day_of_week":
                value = t.weekday()
            elif time_format == 'year':
                value = t.year
            elif time_format == 'day':
                value = t.day
            elif time_format == 'month':
                value = t.month
            elif time_format == 'hour':
                value = t.hour
            elif time_format == 'minutes':
                value = t.minute
            elif time_format == 'seconds':
                value = t.second
            else:
                raise Exception("unknown time grouping: {}".format(time_format))
            result[name] = tf.constant([value], shape=[1, 1])

    def build_input_data(self, timestamp, result=None):
        """
        builds the result dict for the fit or real-time training algoritme.
        :param result: the result value, in case that the callback is used for real-time training.
        :return: a dictionary containing tensors
        """
        feed_dict = {}
        self.fill_timestamp_in_feed_dict(feed_dict,timestamp)  # all thensors need to be built within the session scope, which means within the input builder function.
        for key, asset in self.asset_mappings.iteritems():
            value = asset.value
            if isinstance(value, list):
                length = len(value)
            else:
                length = 1
            self.store_in_feed_dict(feed_dict, asset.column, value, asset.is_sparse, 1, length)
        if result:
            label = tf.constant(feed_dict, shape=[len(result), 1])
            return feed_dict, label
        return feed_dict

    def prepareAssetForInput(self, sensor, column_name, definition, value, is_sparse=False):
        """
        start monitoring changes for the asset and prepares the dict to map an asset id to it's column names
        :param sensor: the sensor object that represents the asset.
        :param column_name: the name of the column/feature that is represented in the model for this asset.
        :param definition: the definition of the feature, so that optional extra parameters can be extracted and stored, like a filter
        :return:
        """
        trigger_filter = None
        if 'trigger' in definition:
            trigger_filter = definition['trigger']
        if sensor.id not in self.asset_mappings:  # store a mapping between name and id, names are stored as single object, asset can only be used 1 time, except for bucketized versions.
            mapping = AssetMapping(column_name, trigger_filter, value, is_sparse)
            self.asset_mappings[sensor.id] = mapping
            registerAssetToMonitor(sensor, self.callback)
        else:
            raise Exception("asset already used as feature. To re-use an asset in a different way (ex: with buckets), use the 'source' keyword")

    def buildAssetFeature(self, definition, name):
        """
        builds a feature based on the definition for an asset.

        supported fields for a feature:
        - asset: the id of the asset to monitor (required)
        - name: the name of the features, must be unique, optional, if ommited, the id will be used
        - buckets: optional, if present, the values will be divided into buckets and the bucket nr will be used instead of the raw value. Only valid when the asset is of type int or number and does not have an enum or labels
        - process original: optional, only valid if buckets is defined: when true, the column with the original values is also included in the model, otherwise, only the buckets version.
        - hash bucket size: optional, only for strings that have no enum or labels. If ommited, 1000 is used.  the size of the hash table
        - embedding vector dimension: optional, when supplied, the feature will be initialzed with a random value and then adjusted during training (see: http://stackoverflow.com/questions/38808643/tf-contrib-layers-embedding-column-from-tensor-flow). This value is the dimension of the embedding vector, ex: 8

        :param definition: a json structure defining how the feature should be built.
        :param name: the name of the feature.
        :return: a tf feature column object.
        """
        id = definition['asset']
        sensor = Sensor(id=id, connection=self.connection)  # we need a sensor object for registering the callbacks, so we can reuse it as much as possible
        profile = sensor.profile
        dataType = profile['type']
        is_sparse = False
        is_bucketized = False
        assetValue = sensor.value

        if 'buckets' in definition:  # user defined buckets in model definition
            column = tf.contrib.layers.real_valued_column(name)
            column = tf.contrib.layers.bucketized_column(column, boundaries=definition['buckets'])
            is_bucketized = True
            # note: don't need to set that it's sparse, this is taken care of by tensor flow, just need to provide the raw value of the asset.
        elif 'hash bucket size' in definition:
            bucketSize = definition['hash bucket size']
            column = tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=bucketSize)
            is_sparse = True
        elif dataType == 'boolean':
            column = tf.contrib.layers.real_valued_column(name)
        elif 'enum' in profile:
            keys = profile['enum']
            column = tf.contrib.layers.sparse_column_with_keys(column_name=name, keys=keys)
            is_sparse = True
        else:
            control = sensor.control
            found = False
            if 'extras' in control:
                extras = control['extras']
                if 'values' in extras:
                    column = tf.contrib.layers.sparse_column_with_keys(column_name=name, keys=extras['values'])
                    is_sparse = True
                    found = True
            if not found:
                if dataType in ['integer', 'number'] \
                        or (dataType == "list"
                            and 'items' in profile
                            and 'type' in profile['items']
                            and profile['items']['type'] in ['number', 'integer']):
                    column = tf.contrib.layers.real_valued_column(name)
                elif dataType == 'string':
                    column = tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000)      # we use a default bucketsize
                    is_sparse = True
                else:
                    raise Exception("unsuported data type, can't use as input for a DNN classifier")
        self.addFeatureColumnToDict(column, definition, name, is_sparse, is_bucketized)
        self.prepareAssetForInput(sensor, name, definition, assetValue, is_sparse)  # start monitoring changes for the asset, store the asset mapping

    def buildTimeFeature(self, definition, name):
        """
        builds a feature based on the definition for a time slot (time of day, day of week, month of year, year,...

        - supported time values: time_of_day, day_of_week, year, day, month, hour, minutes, seconds
        - supported extra params (same meaning as for assets:
            - buckets
            - process original
            - name
            - embedding vector dimension
        :param definition: a json structure defining how the feature should be built.
        :param name: the name of the feature.
        :return: a tf feature column object.
        """
        time_group = definition['time']
        if time_group not in self.time_mappings:
            column = tf.contrib.layers.real_valued_column(name)
            bucketized = False
            if 'buckets' in definition:  # ser defined buckets in model definition
                column = tf.contrib.layers.bucketized_column(column, boundaries=definition['buckets'])
                bucketized = True
            self.addFeatureColumnToDict(column, definition, name, False, bucketized)
            self.time_mappings[time_group] = name
        else:
            raise Exception("time grouping already defined")

    def buildCrossedFeature(self, definition, name):
        """
        builds a crossed column from the feature column names that are provided in the definition.

        - crossed: an array that needs to contain the names of previously declared features
        - hash bucket size: the size of the hash buckets. If not provided, int(1e4) is used.
        - supported extra params (same meaning as for assets:
            - name
            - embedding vector dimension

        :param definition: the crossed feature column definition
        :param name: the name of the feature required.
        :return:
        """
        colNames = definition['crossed']
        cols = []
        for colName in colNames:
            if colName in self.features:
                col = self.features[colName]
                if isinstance(col, tf.contrib.layers.python.layers.feature_column._EmbeddingColumn):        # embedding columns don't work for crossed columns, we need to get the source col.
                    col = col.sparse_id_column
                cols.append(col)
            else:
                raise Exception("{} is not a known feature, can't build crossed feature column".format(colName))
        bucketSize = definition['hash bucket size'] if 'hash bucket size' in definition else int(1e4)
        column = tf.contrib.layers.crossed_column(cols, hash_bucket_size=bucketSize)
        self.addFeatureColumnToDict(column, definition, name, False, True)

    def buildFeatureFromSource(self, definition, name):
        """
        builds a feature, based on a previously declared feature. This is used to declare buckets for an input, where
        you also want to use the original value as an input somehow (ex: deep-wide networks where the waw value is used
          for the wide part, and a bucketized version for the deep part (or visa versa)
        :param definition: the definition of the feature
        :param name: the name of the feature
        :return: none
        """
        source = definition['source']
        if source in self.features:
            source_col = self.features[source]
            if 'buckets' in definition:  # ser defined buckets in model definition
                column = tf.contrib.layers.bucketized_column(source_col, boundaries=definition['buckets'])
            elif 'hash bucket size' in definition:
                bucketSize = definition['hash bucket size']
                column = tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=bucketSize)
            else:
                raise Exception("a 'source' feature needs to have a 'bucket' or 'hash bucket size' definition")
            self.addFeatureColumnToDict(column, definition, name)
        else:
            raise Exception("source feature not found, first declare a feature with the specified name before using a 'source' feature")