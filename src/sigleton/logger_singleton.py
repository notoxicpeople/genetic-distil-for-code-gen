import os
import subprocess
import time

import psutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import config.config as config
from sigleton.singleton import Singleton
from util.counter import Counter
from sigleton.training_process_singleton import TrainingProcessSingleton


class LoggerSingleton(Singleton):
    def __init__(self):
        if not hasattr(self, "initialized"):  # インスタンスが初期化されていないことを確認
            self.initialized = True  # 初期化済みフラグを設定

            singleton: TrainingProcessSingleton = TrainingProcessSingleton()
            self.main_process_itr_counter: Counter = singleton.main_process_itr_counter
            self.training_sessions: list = singleton.training_sessions
            training_params = {
                "config.student_model_name": config.student_model_name,
                "config.teacher_model_name": config.teacher_model_name,
                "config.dataset_name": config.dataset_name,
                "config.eval_input_column": config.eval_input_column,
                "config.eval_test_name_column": config.eval_test_name_column,
                "config.algorithm": config.algorithm,
                "config.use_early_stop": config.use_early_stop,
                "config.total_iterations": config.total_iterations,
                "config.total_models": config.total_models,
                "config.eval_sample_size": config.eval_sample_size,
                "config.output_max_new_tokens": config.output_max_new_tokens,
                "output_max_length": config.input_max_length,
                "config.models_to_replace": config.models_to_replace,
                "config.train_epochs": config.train_epochs,
                "config.train_batches_per_epoch": config.train_batches_per_epoch,
                "config.train_batch_size": config.train_batch_size,
                "config.distillation_temperature": config.distillation_temperature,
                "config.accumulation_steps": config.accumulation_steps,
                "config.max_learning_rate": config.max_learning_rate,
                "config.min_learning_rate": config.min_learning_rate,
                "config.mutation_rate": config.mutation_rate,
                "config.eval_top_p": config.eval_top_p,
                "config.eval_top_k": config.eval_top_k,
                "config.eval_temperature": config.eval_temperature,
                "config.max_grad_norm": config.max_grad_norm,
                "config.log_interval": config.log_interval,
                "config.warmup_steps": config.warmup_steps,
                "config.num_train_optimization_steps": config.num_train_optimization_steps,
                "config.early_stop_count": config.early_stop_count,
                "config.early_stop_loss_change_ratio": config.early_stop_loss_change_ratio,
                "config.alpha_cross_over_hyper_parameter": config.alpha_cross_over_hyper_parameter,
                "config.teach_mode": config.teach_mode,
                "git_commit": get_git_commit_hash(),
                "config.num_samples_per_task_for_humaneval": config.num_samples_per_task_for_humaneval,
                "config.eval_by_exec": config.eval_by_exec,
                "config.mutation_rate_for_hyper_param": config.mutation_rate_for_hyper_param,
            }
            self.tensor_board_global = SummaryWriter(log_dir=os.path.join(config.log_path, singleton.now, "global"))
            self.tensor_board_global.add_text(tag="config/training", text_string=str(training_params), global_step=0)
            self.last_loss = 0
            self.last_loss_distillation = 0
            self.last_loss_classification = 0
            self.last_accuracy = 0
            self.last_log = time.time()
            self.eval_process_total_counter = 0
            self.train_process_counter = 0
            self.total_loss_epoch = 0
            self.train_epoch_total_counter = 0
            self.train_process_total_counter = 0

    # 現状の状況の表示とテキストファイルへの保存を行う
    def log_progress(self, text):
        print(f"{text} | Main process iteration: {self.main_process_itr_counter.value} / {config.total_iterations}")

    # モデル毎の評価のログの記録を行う
    def log_model_eval(self, model, eval_result):
        model["eval_process_counter"] += 1
        logger = model["logger"]
        logger.add_scalar(
            tag="eval",
            scalar_value=eval_result,
            global_step=model["eval_process_counter"],
        )

    # モデル全体の評価のログの記録を行う
    def log_global_eval(self, eval_result):
        self.eval_process_total_counter += 1
        self.tensor_board_global.add_scalar(
            tag="eval",
            scalar_value=eval_result,
            global_step=self.eval_process_total_counter,
        )

    # 学習率のログの記録を行う
    def log_learning_rate(self):
        learning_rate = 0
        for train_session in self.training_sessions:
            learning_rate += train_session["learning_rate"]

        learning_rate /= len(self.training_sessions)

        # 学習率のログの記録
        self.tensor_board_global.add_scalar(
            tag="learning_rate/avg",
            scalar_value=learning_rate,
            global_step=self.main_process_itr_counter.value,
        )

    # 損失の重みのログの記録を行う
    def log_loss_weight(self):
        distillation_weight = 0
        for train_session in self.training_sessions:
            distillation_weight += train_session["distillation_weight"]

        distillation_weight /= len(self.training_sessions)

        # 学習率のログの記録
        self.tensor_board_global.add_scalar(
            tag="weight/avg_distillation_weight",
            scalar_value=distillation_weight,
            global_step=self.main_process_itr_counter.value,
        )

        classification_weight = 0
        for train_session in self.training_sessions:
            classification_weight += train_session["classification_weight"]

        classification_weight /= len(self.training_sessions)

        # 学習率のログの記録
        self.tensor_board_global.add_scalar(
            tag="weight/avg_classification_weight",
            scalar_value=classification_weight,
            global_step=self.main_process_itr_counter.value,
        )

    # 訓練のエポックの終了処理を行うメソッド
    def log_train_epoch(self, train_session):
        # 回数のカウント
        train_session["train_epoch_total_counter"] += 1
        self.train_epoch_total_counter += 1

        # グローバルのログの記録
        self.tensor_board_global.add_scalar(
            tag="epoch/loss",
            scalar_value=self.total_loss_epoch / self.train_process_counter,
            global_step=self.train_epoch_total_counter,
        )

        # モデル毎のログの記録
        logger: SummaryWriter = train_session["logger"]
        train_epoch_total_counter = train_session["train_epoch_total_counter"]
        logger.add_scalar(
            tag="epoch/loss",
            scalar_value=self.total_loss_epoch / self.train_process_counter,
            global_step=train_epoch_total_counter,
        )

        # 初期化
        self.train_process_counter = 0
        self.total_loss_epoch = 0

    # 訓練のログの記録を行う
    def log_train(self, train_session, scheduler, loss, loss_distillation, loss_classification, accuracy):
        # 損失の記録
        self.last_loss = loss.item()
        self.last_loss_distillation = loss_distillation.item()
        self.last_loss_classification = loss_classification.item()
        self.total_loss_epoch += loss.item()
        self.last_accuracy = accuracy

        # 回数のカウント
        self.train_process_counter += 1
        self.train_process_total_counter += 1
        train_session["train_process_total_counter"] += 1

        # ログの記録
        if self.train_process_total_counter % config.log_interval == 0:
            self.log_model_to_tensor_board(train_session, scheduler)
            self.log_global_metrics_to_tensor_board(train_session["model"], scheduler)
            self.last_log = time.time()

    # モデル毎の訓練のログの記録を行う
    def log_model_to_tensor_board(self, train_session, scheduler: torch.optim.lr_scheduler.LambdaLR):
        model: nn.Module = train_session["model"]
        logger: SummaryWriter = train_session["logger"]
        model_iter_count = train_session["train_process_total_counter"]

        for param_name, param in model.named_parameters():
            logger.add_scalar(
                tag="parameter_mean/" + param_name,
                scalar_value=param.data.mean().to(torch.float32),
                global_step=model_iter_count,
            )
            logger.add_scalar(
                tag="parameter_std/" + param_name,
                scalar_value=param.data.std().to(torch.float32),
                global_step=model_iter_count,
            )
            if param.grad is None:
                continue
            logger.add_scalar(
                tag="grad_mean/" + param_name,
                scalar_value=param.grad.data.mean().to(torch.float32),
                global_step=model_iter_count,
            )
            logger.add_scalar(
                tag="grad_std/" + param_name,
                scalar_value=param.grad.data.std().to(torch.float32),
                global_step=model_iter_count,
            )

        logger.add_scalar(
            tag="learning_rate/train_process",
            scalar_value=scheduler.get_lr()[0],
            global_step=model_iter_count,
        )

        logger.add_scalar(
            tag="losses/loss",
            scalar_value=self.last_loss,
            global_step=model_iter_count,
        )
        logger.add_scalar(
            tag="losses/avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.train_process_counter,
            global_step=model_iter_count,
        )
        logger.add_scalar(
            tag="losses/loss_distillation",
            scalar_value=self.last_loss_distillation,
            global_step=model_iter_count,
        )
        logger.add_scalar(
            tag="losses/loss_classification",
            scalar_value=self.last_loss_classification,
            global_step=model_iter_count,
        )
        logger.add_scalar(
            tag="eval/accuracy",
            scalar_value=self.last_accuracy,
            global_step=model_iter_count,
        )

    # テンソルボードへのログの記録を行う
    def log_global_metrics_to_tensor_board(self, student_model: nn.Module, scheduler: torch.optim.lr_scheduler.LambdaLR):
        for param_name, param in student_model.named_parameters():
            self.tensor_board_global.add_scalar(
                tag="parameter_mean/" + param_name,
                scalar_value=param.data.mean().to(torch.float32),
                global_step=self.train_process_total_counter,
            )
            self.tensor_board_global.add_scalar(
                tag="parameter_std/" + param_name,
                scalar_value=param.data.std().to(torch.float32),
                global_step=self.train_process_total_counter,
            )
            if param.grad is None:
                continue
            self.tensor_board_global.add_scalar(
                tag="grad_mean/" + param_name,
                scalar_value=param.grad.data.mean().to(torch.float32),
                global_step=self.train_process_total_counter,
            )
            self.tensor_board_global.add_scalar(
                tag="grad_std/" + param_name,
                scalar_value=param.grad.data.std().to(torch.float32),
                global_step=self.train_process_total_counter,
            )

        self.tensor_board_global.add_scalar(
            tag="losses/loss",
            scalar_value=self.last_loss,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="losses/avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.train_process_counter,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="losses/loss_distillation",
            scalar_value=self.last_loss_distillation,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="losses/loss_classification",
            scalar_value=self.last_loss_classification,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="eval/accuracy",
            scalar_value=self.last_accuracy,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="learning_rate/train_process",
            scalar_value=scheduler.get_lr()[0],
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.train_process_total_counter,
        )
        self.tensor_board_global.add_scalar(
            tag="global/speed",
            scalar_value=time.time() - self.last_log,
            global_step=self.train_process_total_counter,
        )


def get_git_commit_hash() -> str:
    cwd = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
    return out.strip().decode("ascii")
