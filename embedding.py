
class PLMEmb:
    def __init__(self, config):
        self.item_drop_ratio = config['item_drop_ratio']
        self.item_drop_coefficient = config['item_drop_coefficient']

    def __call__(self, dataset, interaction):
        # TODO call method
        pass
