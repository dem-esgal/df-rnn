class AugmentationChain:
    def __init__(self, *ifilters):
        self.ifilters = ifilters

    def __call__(self, x):
        for f in self.ifilters:
            x = f(x)
        return x


def build_augmentations(conf):
    def _chain():
        for cls_name, params in conf:
            cls_f = locals().get(cls_name)
            if cls_f is None:
                raise AttributeError(f'Can not find augmentation for "{cls_name}"')
            yield cls_f(**params)

    return AugmentationChain(*_chain())
