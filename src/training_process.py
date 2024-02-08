import time


import config.config as config
from model.model_eval import ModelEval
from model.model_genrate import ModelGenerate
from model.model_save import ModelSave
from model.model_select import ModelSelect
from model.model_train import ModelTrain
from util.counter import Counter
from sigleton.training_process_singleton import TrainingProcessSingleton
from sigleton.logger_singleton import LoggerSingleton


class TrainingProcess:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.logger: LoggerSingleton = LoggerSingleton()
        self.main_process_itr_counter: Counter = singleton.main_process_itr_counter
        self.start_time = time.time()
        self.early_stop_last_loss = None
        self.early_stop_counter = 0

    def main_process(self):
        self.logger.log_progress("Start main process...")

        # モデルの訓練、選択、生成、保存、評価を行うクラスのインスタンスを生成
        model_train = ModelTrain()
        model_select = ModelSelect()
        model_generate = ModelGenerate()
        model_save = ModelSave()
        model_eval = ModelEval()

        # モデルの初期化
        model_generate.init_train_sessions()

        for i in range(config.total_iterations):
            self.logger.log_progress("Main process start...")

            self.main_process_itr_counter.increment()

            # モデルの訓練
            loss = model_train.train_models()

            # モデルの選択
            model_select.select_models()

            # モデルの削除と生成
            model_generate.remove_and_generate_models()

            # モデルの保存
            model_save.save_model_in_process()

            if config.use_early_stop and self.early_stop(loss):
                # 早期終了の場合は、プロセスを終了
                break

        # モデルの保存
        model_save.save_model_at_end()

        # モデルの評価用データの保存
        model_eval.save_output_for_eval()

        # プロセスの終了時刻を記録
        self.get_process_time()

        self.logger.log_progress("Process finished.")

    # 早期終了を行うかどうかを判定するメソッド
    def early_stop(self, total_loss):
        # 損失の平均値を計算
        loss_mean = total_loss / config.total_models

        if self.early_stop_last_loss is not None:
            loss_change = abs(loss_mean - self.early_stop_last_loss) / max(self.early_stop_last_loss, 1e-10)
            if loss_change > config.early_stop_loss_change_ratio:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= config.early_stop_count:
                    return True

        self.early_stop_last_loss = loss_mean
        return False

    # プロセスの経過時間を表示するメソッド
    def get_process_time(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.logger.log_progress(f"Elapsed time: {elapsed_time} seconds")
