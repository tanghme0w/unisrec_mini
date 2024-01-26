from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from embedding import PLMEmb


class CustomizedTrainDataloader(DataLoader):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, shuffle=False):
        self.dataset = dataset
        self.sample_size = len(dataset)
        self.shuffle = shuffle
        self.transform = PLMEmb(config)
        self.uid_field = dataset.uid_field
        self._init_batch_size_and_step()
        # distributed scenario
        index_sampler = None
        if not config["single_spec"]:
            index_sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
            self.step = max(1, self.step // config["world_size"])
            shuffle = False
        super().__init__(
            dataset=dataset,
            batch_size=self.step,
            collate_fn=self.collate_fn,
            num_workers=config['worker'],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator
        )

    def _init_batch_size_and_step(self):
        batch_size = self.config['train_batch_size']
        # todo: distributed scenario
        self.step = batch_size

    def collate_fn(self, index):
        data = self.dataset[index]
        transformed_data = self.transform(self.dataset, data)
        return transformed_data

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)
