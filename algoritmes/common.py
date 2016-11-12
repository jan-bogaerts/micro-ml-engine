__author__ = 'Jan Bogaerts'
__copyright__ = "Copyright 2016, AllThingsTalk"
__credits__ = []
__maintainer__ = "Jan Bogaerts"
__email__ = "jb@allthingstalk.com"
__status__ = "Prototype"  # "Development", or "Production"


def getFeatureName(feature, default):
    """extract the name of the feature from the definition"""
    if 'name' in feature:
        return feature['name']
    else:
        return default