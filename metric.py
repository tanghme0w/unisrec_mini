import numpy as np
import torch


class TopkMetric():
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """
    metric_need = ["rec.topk"]

    smaller = False

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]
        self.topk = config["topk"]

    def used_info(self, dataobject):
        """Get the bool matrix indicating whether the corresponding item is positive
        and number of positive items for each user.
        """
        rec_mat = dataobject.get("rec.topk")
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        return topk_idx.to(torch.bool).numpy(), pos_len_list.squeeze(-1).numpy()

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = "{}@{}".format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        """Calculate the value of the metric.

        Args:
            pos_index(numpy.ndarray): a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th \
            highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
            pos_len(numpy.ndarray): a vector representing the number of positive items per user, shape of ``(n_users,)``.

        Returns:
            numpy.ndarray: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
        """
        raise NotImplementedError(
            "Method [metric_info] of top-k metric should be implemented."
        )


class Hit(TopkMetric):
    r"""HR_ (also known as truncated Hit-Ratio) is a way of calculating how many 'hits'
    you have in an n-sized list of ranked items. If there is at least one item that falls in the ground-truth set,
    we call it a hit.

    .. _HR: https://medium.com/@rishabhbhatia315/recommendation-system-evaluation-metrics-3f6739288870

    .. math::
        \mathrm {HR@K} = \frac{1}{|U|}\sum_{u \in U} \delta(\hat{R}(u) \cap R(u) \neq \emptyset),

    :math:`\delta(·)` is an indicator function. :math:`\delta(b)` = 1 if :math:`b` is true and 0 otherwise.
    :math:`\emptyset` denotes the empty set.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, _ = self.used_info(dataobject)
        result = self.metric_info(pos_index)
        metric_dict = self.topk_result("hit", result)
        return metric_dict

    def metric_info(self, pos_index):
        result = np.cumsum(pos_index, axis=1)
        return (result > 0).astype(int)


class NDCG(TopkMetric):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality,
    where positions are discounted logarithmically. It accounts for the position of the hit by assigning
    higher scores to hits at top ranks.

    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \mathrm {NDCG@K} = \frac{1}{|U|}\sum_{u \in U} (\frac{1}{\sum_{i=1}^{\min (|R(u)|, K)}
        \frac{1}{\log _{2}(i+1)}} \sum_{i=1}^{K} \delta(i \in R(u)) \frac{1}{\log _{2}(i+1)})

    :math:`\delta(·)` is an indicator function.
    """

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pos_index, pos_len = self.used_info(dataobject)
        result = self.metric_info(pos_index, pos_len)
        metric_dict = self.topk_result("ndcg", result)
        return metric_dict

    def metric_info(self, pos_index, pos_len):
        len_rank = np.full_like(pos_len, pos_index.shape[1])
        idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

        iranks = np.zeros_like(pos_index, dtype=np.float)
        iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
        for row, idx in enumerate(idcg_len):
            idcg[row, idx:] = idcg[row, idx - 1]

        ranks = np.zeros_like(pos_index, dtype=np.float)
        ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
        dcg = 1.0 / np.log2(ranks + 1)
        dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

        result = dcg / idcg
        return result
