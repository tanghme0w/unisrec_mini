class Config:
    def __init__(
            self, model=None, dataset=None, config_file_list=None, config_dict=None
    ):
        self.config = dict()
        self.config['data_path'] = "small_scale_test/test_interaction.jsonl"
        self.config['worker'] = 0
        self.config['single_spec'] = True
        self.config['mmap_idx_path'] = 'small_scale_test/mmap_idx_100'
        self.config['mmap_idx_shape'] = (100,)
        self.config['mmap_emb_shape'] = (100, 768)
        self.config['mmap_emb_path'] = 'small_scale_test/mmap_data_100_768'
        self.config['device'] = 'cpu'
        self.config['train_batch_size'] = 5
        # model hyperparameters
        self.config["n_layers"] = 2
        self.config["n_heads"] = 2
        self.config["hidden_size"] = 300  # same as embedding_size
        self.config["inner_size"] = 256  # the dimensionality in feed-forward layer
        self.config["hidden_dropout_prob"] = 0.5
        self.config["attn_dropout_prob"] = 0.5
        self.config["hidden_act"] = 'gelu'
        self.config["layer_norm_eps"] = 1e-12
        self.config["initializer_range"] = 0.02
        self.config["loss_type"] = 'CE'
        # training hyperparams
        self.config['learning_rate'] = 0.001
        self.config['weight_decay'] = 0.
        self.config['pretrain_epochs'] = 10
        self.config['checkpoint_dir'] = 'saved'
        self.config['save_step'] = 5
        self.config['train_stage'] = 'pretrain'
        self.config['temperature'] = 0.07
        self.config['lambda'] = 0.001
        # MoE hyperparams
        self.config['n_exps'] = 8
        self.config['adaptor_layers'] = [768, 300]
        self.config['adaptor_dropout_prob'] = 0.2
        # dataset hyperparameters
        self.config["MAX_ITEM_LIST_LENGTH"] = 50
        # field names (adapt to RecBole)
        self.config["USER_ID_FIELD"] = "user_id"
        self.config["ITEM_ID_FIELD"] = "item_id"
        self.config["LABEL_FIELD"] = "label"
        self.config["TIME_FIELD"] = "timestamp"
        self.config["LIST_SUFFIX"] = "_list"
        self.config["ITEM_LIST_LENGTH_FIELD"] = "item_length"
        self.config["NEG_PREFIX"] = "neg_"

    def __getitem__(self, item):
        return self.config[item]
