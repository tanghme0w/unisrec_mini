import json

from torch.utils.data import Dataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import torch
from interaction import Interaction


class PretrainDataset(Dataset):
    def __init__(self, config):
        # get maximum interaction sequence length
        self.max_inter_length = config["MAX_ITEM_LIST_LENGTH"]
        # get dataset size
        # self.dataset_size = config['dataset_size']
        # load data from file
        self._load_inter_feat(config['data_path'])
        # do preprocessing
        self._data_preprocessing()

    def __len__(self):
        return len(self.inter_feat)

    def __getitem__(self, index):
        return self.inter_feat[index]

    # adapt to RecBole framework, has no effect
    def num(self, key):
        return 1

    def _data_preprocessing(self):
        pass

    def _load_inter_feat(self, ds_path):
        data = []
        with open(ds_path) as fp:
            for line in fp:
                entry = json.loads(line)
                user_sequence = entry["user_sequence"]
                # strip leading zeros
                user_sequence = [x for x in user_sequence if x != 0]
                entry["item_id_list"] = user_sequence[:-1]
                # get target item
                entry["item_id"] = user_sequence[-1]
                entry["item_length"] = len(entry["item_id_list"])
                entry.pop("user_sequence", None)
                data.append(entry)

        # convert to dataframe
        df = pd.DataFrame(data)
        # convert to Interaction type
        self.inter_feat = self._dataframe_to_interaction(df)

    def _dataframe_to_interaction(self, data):
        new_data = {}
        for key in data:
            value = data[key].values
            if key in ["item_id", "item_length", "user_id"]:
                new_data[key] = torch.LongTensor(value)
            elif key == "item_id_list":
                seq_data = [torch.LongTensor(d[: self.max_inter_length]) for d in value]
                new_data[key] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)
