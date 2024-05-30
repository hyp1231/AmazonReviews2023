import importlib
from recbole.utils import get_model as get_recbole_model
from recbole.data.utils import create_dataset as create_recbole_dataset


def get_model(model_name):
    model_file_name = model_name.lower()
    module_path = f'model.{model_file_name}'
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
        model_class = getattr(model_module, model_name)
    else:
        try:
            model_class = get_recbole_model(model_name)
        except:
            raise ValueError('`model_name` [{}] is not the name of an existing model or RecBole models.'.format(model_name))
    return model_class


def create_dataset(config):
    dataset_module = importlib.import_module('data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        return create_recbole_dataset(config)
