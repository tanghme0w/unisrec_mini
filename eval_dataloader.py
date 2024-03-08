"""
copied and edited from recbole.data.dataloader.general_dataloader
################################################
"""

import numpy as np
import torch
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.sampler.sampler import RepeatableSampler


class FullSortEvalDataLoader(AbstractDataLoader):
    """:class:`FullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, shuffle=False):
        # Haomiao Tang
        sampler = RepeatableSampler(
            ["train", "valid", "test"],
            dataset,
            "uniform",
            1.0
        )
        sampler.set_phase("valid")

        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field

        self.sample_size = len(dataset)
        if shuffle:
            self.logger.warnning("FullSortEvalDataLoader can't shuffle")
            shuffle = False
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        self.uid2positive_item[uid] = torch.tensor(
            list(positive_item), dtype=torch.int64
        )
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(list(history_item), dtype=torch.int64)

    def _init_batch_size_and_step(self):
        batch_size = self.config["eval_batch_size"]
        self.step = batch_size
        self.set_batch_size(batch_size)

    def update_config(self, config):
        super().update_config(config)

    def collate_fn(self, index):
        index = np.array(index)

        interaction = self._dataset[index]
        transformed_interaction = self.transform(self._dataset, interaction)
        inter_num = len(transformed_interaction)
        positive_u = torch.arange(inter_num)
        positive_i = transformed_interaction[self.iid_field]

        return transformed_interaction, None, positive_u, positive_i


# testing
if __name__ == '__main__':
    from config import Config
    from dataset import PretrainDataset
    config = Config()
    ds = PretrainDataset(config)
    valid_dataloader = FullSortEvalDataLoader(config, ds)
