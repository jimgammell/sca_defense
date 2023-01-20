import importlib
import os
import torch

def getattr_from_file(attr_name):
    modules = []
    for package in ['datasets', 'models', 'training']:
        for module_filename in os.listdir(os.path.join('.', package)):
            if module_filename in ['__init__.py', '__pycache__', '.ipynb_checkpoints']:
                continue
            module = importlib.import_module('.'+module_filename.split('.')[0], package)
            modules.append(module)
    modules.extend([torch.nn, torch.optim])
    for module in modules:
        try:
            attr = getattr(module, attr_name)
            return attr
        except:
            pass
    raise Exception('Could not find attribute called \'{}\' in the searched modules [\n\t{}]'.format(
        attr_name, ',\n\t'.join(str(module) for module in modules)))