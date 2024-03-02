from metric import Hit, NDCG
import torch
import numpy as np

topk = [10, 50]

class Evaluator:
    def __init__(self, model):
        self.model = model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        pass

    def eval_func(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        # Note: interaction without item ids
        scores = self.model.full_sort_predict(interaction.to(self.device))
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

_, scores_tensor, positive_u, positive_i = eval_func(batched_data)

_, topk_idx = torch.topk(scores_tensor, max(topk), dim=-1)  # n_users x k
pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
pos_matrix[positive_u, positive_i] = 1
pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
data_struct = torch.cat((pos_idx, pos_len_list), dim=1)