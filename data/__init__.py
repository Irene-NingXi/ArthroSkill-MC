"""Dataset APIs with optional training dependencies loaded on demand."""
__all__ = ['ArthroscopyDataset', 'collate_fn']


def __getattr__(name):
    if name in __all__:
        from .dataset import ArthroscopyDataset, collate_fn
        return {'ArthroscopyDataset': ArthroscopyDataset, 'collate_fn': collate_fn}[name]
    raise AttributeError(name)
