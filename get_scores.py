import torch

from config import Config
from dataset import PretrainDataset
from missrec import MISSRec


def get_scores(ckpt):
    # load model
    config = Config()
    data = PretrainDataset(config)
    model = MISSRec(config, data)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    return model.full_sort_predict(data.inter_feat)


if __name__ == "__main__":
    scores = get_scores("/home/me0w/Desktop/hw_rec/unisrec_mini/saved/MISSRec-news-10.pth")
    print(scores)
