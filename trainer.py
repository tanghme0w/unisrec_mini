import os
from time import time

import torch
from torch import optim
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.start_epoch = 0
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.device = config['device']
        self.pretrain_epochs = config['pretrain_epochs']
        self.checkpoint_dir = config['checkpoint_dir']
        self.save_step = config['save_step']
        self.train_loss_dict = dict()


class PretrainTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.saved_model_file = None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=f"Train {epoch_idx}"
            )
            if show_progress
            else train_data
        )

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            loss = loss_func(interaction)
            total_loss = loss.item() if total_loss is None else total_loss + loss.item()
            if torch.isnan(loss):
                raise ValueError("Training loss is nan")
            loss.backward()
            self.optimizer.step()

        return total_loss

    def pretrain(self, train_data, show_progress=False):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = train_loss
            training_end_time = time()
            print(f"thm debug - epoch: {epoch_idx}. training loss: {train_loss}. time: {training_end_time - training_start_time}")

            # save model
            if (epoch_idx + 1) % self.save_step == 0:
                save_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}.pth".format(
                        "MISSRec", "news", str(epoch_idx + 1)
                    )
                )
                self.save_pretrained_model(epoch_idx, save_model_file)

    def save_pretrained_model(self, epoch, saved_model_file):
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "other_parameter": self.model.other_parameter(),
        }
        torch.save(state, saved_model_file)

    def evaluate(self, eval_data, show_progress):
        pass
        # load model: best or current model
        # TODO
        # set model to eval mode
        self.model.eval()
        # build the eval dataloader as FullSortEvalDataLoader
        # TODO
        eval_func = self._full_sort_batch_eval
        if self.item_tensor is None:
            # item_tensor: all item ids
            self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        self.tot_item_num = eval_data._dataset.item_num
        iter_data = (
            tqdm(
                eval_data,
                desc="evaluate"
            )
            if show_progress
            else eval_data
        )
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        if not self.config["single_spec"]:
            result = self._map_reduce(result, num_sample)
        return result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i


class DDPPretrainTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
        self.pretrain_epochs = config['pretrain_epochs']

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=f"Train {epoch_idx:>5}"
            )
            if show_progress
            else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            interaction = self._trans_dataload(interaction)

    def pretrain(self, train_data, show_progress):
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0 and self.lrank == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    '{}-{}-{}.pth'.format(self.config['model'], self.config['dataset'], str(epoch_idx + 1))
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)
