from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, config):
        self.dataset_size = config['dataset_size']
        # load data from file
        self._load_inter_feat(config['data_path'])
        # do preprocessing
        self._data_preprocessing()

    def __len__(self):
        return len(self.inter_feat)

    def __getitem__(self, index):
        return self.inter_feat[index]

    def _data_preprocessing(self):
        pass

    def _load_inter_feat(self, ds_path):
        
        self.inter_feat = None
