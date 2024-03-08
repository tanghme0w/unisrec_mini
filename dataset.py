import json
import os

from torch.utils.data import Dataset
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
import torch
from interaction import Interaction
from tqdm import tqdm
from interaction import cat_interactions


class PretrainDataset(Dataset):
    def __init__(self, config):
        # get maximum interaction sequence length
        self.max_inter_length = config["MAX_ITEM_LIST_LENGTH"]
        # load data files
        self.data_path = config['data_path']
        self._load_data()
        # do preprocessing
        self._data_preprocessing()  # has no effect for now

        # to be compatible with recbole.sampler.sampler.RepeatableSampler
        self.uid_field = "user_id"
        self.iid_field = "item_id"
        self.user_num = self.dataset_size
        self.item_num = config['mmap_idx_shape'][0]

    def _load_data(self):
        self.data_files = []

        # collect all data files if input path is a directory
        if os.path.isdir(self.data_path):
            for file in os.listdir(self.data_path):
                file_path = os.path.join(self.data_path, file)
                # only collect files more than 10KB
                if os.path.getsize(file_path) >= 10240 and file_path.endswith('.jsonl'):
                    self.data_files.append(file_path)
        else:
            self.data_files.append(self.data_path)

        # count total size
        entry_count = 0
        for file in tqdm(self.data_files, desc="Count File Entries"):
            with open(file) as fp:
                entry_count += len(fp.readlines())
        self.dataset_size = entry_count
        if self.dataset_size == 0:
            raise ValueError("Dataset is empty. Please check data path configuration.")

        # establish iterator
        self.data_file_iterator = iter(self.data_files)

        # read first data file
        self.current_opening_file = self.data_file_iterator.__next__()
        self.current_offset = 0
        self._load_inter_feat(self.current_opening_file)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # if data index exceeds the last index of current opening file
        if index[-1] >= self.last_entry:
            ret_data = []
            # split indices
            cf_idx = [i for i in index if i < self.last_entry]  # index belonging to current file
            nf_idx = [i for i in index if i >= self.last_entry] # index belonging to next file
            # read remaining data of current file
            cf_inter = self.inter_feat[[idx - self.current_offset for idx in cf_idx]]
            # load next data file
            self.current_offset = self.last_entry
            self._load_inter_feat(self.data_file_iterator.__next__())
            # read data of next file
            nf_inter = self.inter_feat[[idx - self.current_offset for idx in nf_idx]]
            # concatenate interactions
            return cat_interactions([cf_inter, nf_inter])
        else:
            return self.inter_feat[[idx - self.current_offset for idx in index]]

    # adapt to RecBole framework, has no effect
    def num(self, key):
        return 1

    def _data_preprocessing(self):
        pass

    def _load_inter_feat(self, ds_path):
        data = []
        with open(ds_path) as fp:
            # set counters
            lines = fp.readlines()
            self.last_entry = self.current_offset + len(lines)
            # read data
            for line in lines:
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
