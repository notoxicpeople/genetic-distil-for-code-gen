from datetime import datetime

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

import config.config as config
from sigleton.singleton import Singleton
from util.counter import Counter


class TrainingProcessSingleton(Singleton):
    def __init__(self):
        if not hasattr(self, "initialized"):  # インスタンスが初期化されていないことを確認
            self.initialized = True  # 初期化済みフラグを設定

            # デバッグモードの場合は、データセットを小さくする
            if config.algorithm == config.algorithm.DEBUG:
                config.train_batch_size = 1
                config.train_epochs = 1
                config.total_models = 3
                config.models_to_replace = 1
                config.total_iterations = 1
                config.eval_sample_size = 1
                config.train_batches_per_epoch = 1
                config.log_interval = 1

            # 並列処理の初期化
            self.accelerator = Accelerator()

            # メインプロセスの繰り返し回数のカウント
            self.main_process_itr_counter = Counter()

            # CUDA利用可能かチェック
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.torch_dtype = torch_dtype
            # self.torch_dtype = torch.float32

            # トークナイザの初期化
            student_tokenizer = AutoTokenizer.from_pretrained(
                config.student_model_name, torch_dtype=torch_dtype, cache_dir="cache/tokenizer/" + config.student_model_name, trust_remote_code=True, padding_side="left"
            )
            student_tokenizer.pad_token = student_tokenizer.eos_token
            self.student_tokenizer = student_tokenizer

            now = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.now = now

            # モデルの初期化
            self.training_sessions = []
