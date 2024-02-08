from accelerate import Accelerator

import config.config as config
from sigleton.training_process_singleton import TrainingProcessSingleton


class ModelSave:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.training_sessions: list = singleton.training_sessions
        self.accelerator: Accelerator = singleton.accelerator
        self.now = singleton.now
        # モデルが生存している最長のプロセス回数
        self.longest_suvival_process_counter = 0

    def save_model_in_process(self):
        for training_session in self.training_sessions:
            # 生存プロセス回数を更新
            training_session["suvival_process_counter"] += 1

            # 生存プロセス回数が最長の場合は、モデルを保存
            suvival_process_counter = training_session["suvival_process_counter"]
            if suvival_process_counter > self.longest_suvival_process_counter:
                self.accelerator.save_model(training_session["model"], f"{config.save_model_in_process_path}/{self.now}/{training_session['id']}")

                # 生存プロセス回数を更新
                self.longest_suvival_process_counter = suvival_process_counter

    def save_model_at_end(self):
        for i, training_session in enumerate(self.training_sessions):
            rank = i + 1
            id = training_session["id"]
            self.accelerator.save_model(training_session["model"], f"{config.save_model_at_end_path}/{self.now}/{rank}_{id}")
