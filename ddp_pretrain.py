import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from config import Config
from trainer import DDPPretrainTrainer
from missrec import MISSRec


def pretrain(rank, world_size, dataset, **kwargs):

    pretrain_data = None
    config = Config()

    model = MISSRec(config, dataset).to(config['device'])

    trainer = DDPPretrainTrainer(config, model)

    # model pre-training
    trainer.pretrain(pretrain_data, show_progress=(rank == 0))

    dist.destroy_process_group()

    return config['model'], config['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, default='12355', help='port for ddp')
    parser.add_argument('-d', type=str, default='FHCKM', help='dataset name')
    args, _ = parser.parse_known_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}."
    world_size = n_gpus

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.p

    mp.spawn(pretrain,
             args=(world_size, args.d,),
             nprocs=world_size,
             join=True)
