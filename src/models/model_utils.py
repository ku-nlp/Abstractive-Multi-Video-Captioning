import importlib


IMPORT_TABLE = {
    'transformer-multi':
        'models.transformer_multiple_input_model',
    'transformer-single':
        'models.transformer_single_input_model'
}


def get_model(name, cfg):
    if name in IMPORT_TABLE:
        module = IMPORT_TABLE[name]
        generator = importlib.import_module(module)
        return generator.generate_model(**cfg)
    else:
        raise ImportError
