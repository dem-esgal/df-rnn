import functools
import operator


def nested_list_to_flat_with_collate(collate_fn):
    def _func(batch):
        batch = functools.reduce(operator.iadd, batch)
        return collate_fn(batch)

    return _func
