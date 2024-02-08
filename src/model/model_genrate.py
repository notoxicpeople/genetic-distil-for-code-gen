import copy
import os
import random
import uuid

import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM

import config.config as config
from util.counter import Counter
from sigleton.training_process_singleton import TrainingProcessSingleton
from config.config import Algorithm
from sigleton.logger_singleton import LoggerSingleton


class ModelGenerate:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.logger: LoggerSingleton = LoggerSingleton()
        self.training_sessions: list = singleton.training_sessions
        self.torch_dtype = singleton.torch_dtype
        self.accelerator: Accelerator = singleton.accelerator
        self.now = singleton.now
        self.main_process_itr_counter: Counter = singleton.main_process_itr_counter

    def init_train_sessions(self):
        for _ in range(config.total_models):
            train_session = self.generate_train_session()
            self.training_sessions.append(train_session)

    # 最も精度が低いモデルを削除し、新しいモデルを追加するメソッド
    def remove_and_generate_models(self):
        self.logger.log_progress("Start remove_and_generate_train_sessions...")

        for _ in range(config.models_to_replace):
            # 最も精度が低いモデルを削除
            self.training_sessions.pop()

        for _ in range(config.models_to_replace):
            # 新しいモデルを追加
            train_session = self.generate_train_session(isFirstGeneration=False)
            self.training_sessions.append(train_session)

        # 学習率のログの記録
        self.logger.log_learning_rate()

        # 蒸留の重みのログの記録
        self.logger.log_loss_weight()

    # 生徒モデルの作成
    def generate_train_session(self, isFirstGeneration=True):
        self.logger.log_progress("Start generate_train_session...")

        if isFirstGeneration:
            # 初回の場合は、ランダムにモデルを作成
            model = AutoModelForCausalLM.from_pretrained(config.student_model_name, trust_remote_code=True, torch_dtype=self.torch_dtype, device_map="auto", cache_dir=config.student_model_cache_path)
            learning_rate = random.uniform(config.min_learning_rate, config.max_learning_rate)
            distillation_weight = random.uniform(0.3, 0.8)
            classification_weight = random.uniform(0.3, 0.8)
        else:
            # 2世代目以降の場合は、精度の高いモデルを選択
            top_models = self.training_sessions[:2]

            # 精度の高いモデルを交叉
            if config.algorithm == Algorithm.GENETIC_ALGORITHM_UNIFORM or config.algorithm == Algorithm.DEBUG:
                model = self.cross_over_model_uniform(top_models[0]["model"], top_models[1]["model"])
            elif config.algorithm == Algorithm.GENETIC_ALGORITHM_ARITHMETIC:
                model = self.cross_over_model_arithmetic(top_models[0]["model"], top_models[1]["model"])

            # 学習率を交叉
            learning_rate = self.cross_over_hyper_parameters(top_models[0]["learning_rate"], top_models[1]["learning_rate"])

            # 蒸留の重みを交叉
            distillation_weight = self.cross_over_hyper_parameters(top_models[0]["distillation_weight"], top_models[1]["distillation_weight"])

            # 分類の重みを交叉
            classification_weight = self.cross_over_hyper_parameters(top_models[0]["classification_weight"], top_models[1]["classification_weight"])

        # WTEの重みを固定。
        model.transformer.wte.weight.requires_grad = False

        # オプティマイザーの初期化
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        if config.use_cpu:
            model = model.to("cpu")
            optimizer = self.accelerator.prepare(optimizer)
        else:
            model, optimizer = self.accelerator.prepare(model, optimizer)

        # TensorBoardの初期化
        # モデルIDは世代を示す値とUUIDを組み合わせたもの。生成するモデルは次世代のものなので、メインプロセスの繰り返し回数に1を足す。
        model_id = f"{self.main_process_itr_counter.value + 1}-{uuid.uuid4()}"
        tensor_board = SummaryWriter(log_dir=os.path.join(config.log_path, self.now, "models", str(model_id)))

        return {
            "id": model_id,
            "model": model,
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "distillation_weight": distillation_weight,
            "classification_weight": classification_weight,
            "logger": tensor_board,
            "eval_process_counter": 0,
            "train_process_total_counter": 0,
            "train_epoch_total_counter": 0,
            "suvival_process_counter": 0,
        }

    def cross_over_model_uniform(self, model1, model2):
        # model1のコピーを作成
        new_model = copy.deepcopy(model1)

        # 各パラメータに対して交叉と突然変異を適用
        for (name1, param1), (name2, param2) in zip(new_model.named_parameters(), model2.named_parameters()):
            if random.random() < 0.5:
                param1.data.copy_(param2.data)
            # 突然変異の適用
            if random.random() < config.mutation_rate:
                # 突然変異の量を決定
                mutation = 0.01 * torch.randn_like(param1.data)
                param1.data.add_(mutation)

        return new_model

    def cross_over_model_arithmetic(self, model1, model2):
        # model1のコピーを作成
        new_model = copy.deepcopy(model1)

        # 各パラメータに対して交叉と突然変異を適用
        for (name1, param1), (name2, param2) in zip(new_model.named_parameters(), model2.named_parameters()):
            # 算術交叉（平均値）の適用
            param1.data.copy_(0.5 * (param1.data + param2.data))

            # 突然変異の適用
            if random.random() < config.mutation_rate:
                # 突然変異の量を決定
                mutation = 0.01 * torch.randn_like(param1.data)
                param1.data.add_(mutation)

        return new_model

    def cross_over_hyper_parameters(self, param1, param2):
        # ハイパーパラメータの最小値と最大値を求める
        min_param = min(param1, param2)
        max_param = max(param1, param2)

        # 拡張された範囲を計算
        lower_bound = min_param - config.alpha_cross_over_hyper_parameter * (max_param - min_param)
        upper_bound = max_param + config.alpha_cross_over_hyper_parameter * (max_param - min_param)

        # 拡張された範囲から新しいハイパーパラメータをランダムに選択
        new_param = random.uniform(lower_bound, upper_bound)

        if random.random() < config.mutation_rate_for_hyper_param:
            # ガウス分布を用いた突然変異
            mutation_amount = random.gauss(0, 1) * 0.1 * (max_param - min_param)
            new_param = new_param + mutation_amount

        # ハイパーパラメータの最小値を設定
        new_param = max(new_param, 1e-10)

        return new_param
