class Config:
    def __init__(
            self, model=None, dataset=None, config_file_list=None, config_dict=None
    ):
        self.config = dict()
        self.config['dataset'] = None
        self.config['data_path'] = None
        self.config['worker'] = 0

    def __getitem__(self, item):
        return self.config[item]