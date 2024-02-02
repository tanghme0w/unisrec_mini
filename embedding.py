import numpy as np
import torch

from interaction import Interaction


class PLMEmb:
    def __init__(self, config):
        self.mmap_idx_path = config['mmap_idx_path']
        self.mmap_emb_path = config['mmap_emb_path']
        self.mmap_idx_shape = config['mmap_idx_shape']
        self.mmap_emb_shape = config['mmap_emb_shape']

    def __call__(self, dataset, interaction):
        item_seq = interaction['item_id_list']
        pos_item = interaction['item_id']
        idx_mmap = np.memmap(self.mmap_idx_path, dtype=np.int32, shape=self.mmap_idx_shape)
        data_mmap = np.memmap(self.mmap_emb_path, dtype=np.float32, shape=self.mmap_emb_shape)
        item_emb_seq = torch.Tensor(data_mmap[idx_mmap[item_seq]])
        pos_item_emb = torch.Tensor(data_mmap[idx_mmap[pos_item]])
        interaction.update(Interaction(
            {
                'item_emb_list': item_emb_seq,
                'pos_item_emb': pos_item_emb
            }
        ))
        return interaction
