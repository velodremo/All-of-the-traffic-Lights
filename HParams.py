class HParams():
    def __init__(self, **kwargs):
        self.set_params(kwargs)

    def set_params(self, params_dict):
        for k,v in params_dict.items():
            assert type(k) == str
            setattr(self, k, v)


global_hparams = HParams(OBSERVED_RADIUS= 50)
