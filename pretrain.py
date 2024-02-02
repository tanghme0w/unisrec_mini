import argparse
from config import Config
from dataset import PretrainDataset
from missrec import MISSRec
from dataloader import CustomizedTrainDataloader
from trainer import PretrainTrainer


def pretrain(dataset, **kwargs):
    # config includes dataset config & model config
    config = Config()

    # dataset
    dataset = PretrainDataset(config)
    dataloader = CustomizedTrainDataloader(config, dataset, shuffle=False)

    # model
    model = MISSRec(config, dataset).to(config['device'])

    # trainer
    trainer = PretrainTrainer(config, model)
    trainer.pretrain(dataloader, show_progress=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    pretrain(args.d)
