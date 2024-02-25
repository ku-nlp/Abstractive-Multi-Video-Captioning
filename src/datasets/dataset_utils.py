import importlib


_IMPORT_TABLE = {
    'abstracts':
        'datasets.abstracts_video',
    'abstracts-gold':
        'datasets.abstracts_gold',
    'abstracts-pred':
        'datasets.abstracts_pred',
    'abstracts-gold-t5':
        'datasets.abstracts_gold_t5',
    'abstracts-pred-t5':
        'datasets.abstracts_pred_t5',
    'vatex':
        'datasets.vatex'
}


def get_dataset(name, split_type, cfg):
    if name in _IMPORT_TABLE:
        module = _IMPORT_TABLE[name]
        generator = importlib.import_module(module)
        return generator.generate_dataset(split_type=split_type, **cfg)
    else:
        raise ImportError
