from . import BaseModelTraining, BaseModelCustomTraining

class BaseModel(object):
    def __new__(cls, config_specs):
        if config_specs['custom_training']:
            return BaseModelCustomTraining(config_specs )
        else :
            return BaseModelTraining(config_specs)

