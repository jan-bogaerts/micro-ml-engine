__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"

import logging
import logging.config
logging.config.fileConfig('config/logging.config')
from flask import Flask, render_template, Response, request
from flask.ext.api import status
import os
import json
import numpy.random as nprnd
import uuid

import att_trusted_event_server.iotApplication as iotApp
from att_event_engine.att import Client
import config.config as config
import algoritmes

app = Flask(__name__)
iot = iotApp.IotApplication(config.UserName, config.Pwd, config.Api, config.Broker, "ml")


def prepareRandoms(definition, columns):
    """
    prepares the training data declared in definition.
    loads it if required and adds records, depending on randomisation definition.
    :param definition: the definition for the data: an array of arrays
    :param columns: the columns in the data, used for the randomization
    :return:
    """
    values = definition['values']
    if 'randomize' in definition:
        for random in definition['randomize']:
            rg = random['range']
            if 'count' not in random:
                raise Exception("invalid random definition: missing 'count' field to identify how many random records need to be generated")
            randomVals = nprnd.randint(rg[0], rg[1], random['count'])
            newValues = list(values)            # make a ocopy of the current
            fieldIndex = columns.index(random['field'])
            for random in randomVals:
                for row in newValues:
                    newRow = list(row)                  # make a duplicate of the row, so we don't modify the original row.
                    newRow[fieldIndex] = random
                    values.append(newRow)
    return values




def load(definition, name, train=False):
    """
    load the model
    :param definition: the json object defining the model
    :param train: true if the model should be trained with the data in the definition
    :return:
    """
    connection = Client()
    connection.connect(definition['username'], definition['pwd'])
    obj = algoritmes.create(definition, connection, name)
    if train and 'traindata' in definition:               # if there is traindata specified, load it
        traindata = definition['traindata']
        if not 'columns' in traindata:
            raise Exception("missing 'columns' field in training data")
        if not 'groups' in traindata:
            raise Exception("missing 'groups' field in training data")
        if not 'steps' in traindata:
            raise Exception("missing 'steps' field in training data")
        columns = traindata['columns']
        steps = traindata['steps']
        for group in traindata['groups']:
            data = prepareRandoms(group, columns)
            obj.train(data, columns, steps)
    return obj

def loadAll():
    """
    loads al the known statistics defs from disc and registers them to monitor for incomming events
    :return:
    """
    files = [f for f in os.listdir('definitions') if os.path.isfile(os.path.join('definitions', f))]
    for file in files:
        with open(os.path.join('definitions', file)) as fp:
            try:
                data = json.load(fp)
                #todo: turn training off after debugging
                load(data, os.path.splitext(file)[0], False)
            except:
                logging.exception("failed to load model: {0}".format(file))

def storeDef(name, value):
    """
    stores the definition on disk
    :param name: the name to use
    :param value: the value (string)
    :return:None
    """
    with open(os.path.join('definitions', name), 'w') as f:
        f.write(value)


@app.route('/definition', methods=['POST'])
def addEvent():
    """
    called when a statistics definition needs to be added to the list
    :return: ok or error
    """
    try:
        id = uuid.uuid4()
        data = json.loads(request.data)
        obj = load(data, id, True)
        storeDef(id, request.data)
        return id, status.HTTP_200_OK
    except Exception as e:
        logging.exception("failed to store definition")
        return str(e), status.HTTP_409_CONFLICT

@app.route('/definition/<id>', methods=['PUT'])
def updateEvent(id):
    """
    called when a statistics definition needs to be added to the list
    :return: ok or error
    """
    try:
        data = json.loads(request.data)
        obj = load(data, id, True)
        storeDef(id, request.data)
        return id, status.HTTP_200_OK
    except Exception as e:
        logging.exception("failed to store definition")
        return str(e), status.HTTP_409_CONFLICT

try:
    loadAll()
    iot.run()
    if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, threaded=True, port=config.port, use_reloader=False)  # blocking
except:
    logging.exception("failed to start ML engine")