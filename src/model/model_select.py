import random

from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import config.config as config
from sigleton.training_process_singleton import TrainingProcessSingleton
from util.code_extract import extract_python_code
from util.context_manager import swallow_io, time_limit
from sigleton.logger_singleton import LoggerSingleton


class ModelSelect:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.training_sessions: list = singleton.training_sessions
        self.accelerator: Accelerator = singleton.accelerator
        self.logger: LoggerSingleton = LoggerSingleton()
        self.student_tokenizer: AutoTokenizer = singleton.student_tokenizer

        # 評価データのロード
        code_dataset_eval = load_dataset(config.eval_dataset_name, "sanitized")
        self.dataset_eval = list(code_dataset_eval["train"])
        self.dataset_eval.extend(list(code_dataset_eval["test"]))
        self.dataset_eval.extend(list(code_dataset_eval["validation"]))
        self.dataset_eval.extend(list(code_dataset_eval["prompt"]))

    def select_models(self):
        self.logger.log_progress("Start select_models...")

        # 評価対象の検証データをランダムに選択
        picked_eval = random.sample(self.dataset_eval, config.eval_sample_size)

        # 各モデルの精度を計算
        if config.eval_by_exec:
            accuracies = [self.calculate_fitness(training_session, picked_eval) for training_session in self.training_sessions]
        else:
            accuracies = [model["accuracy"] for model in self.training_sessions]

        # モデルと精度をペアにする
        model_accuracy_pairs = zip(self.training_sessions, accuracies)

        # 精度順にソート
        sorted_pairs = sorted(model_accuracy_pairs, key=lambda x: x[1], reverse=True)

        # モデルのリストを更新
        self.training_sessions = [model for model, _ in sorted_pairs]

        # 精度の平均値を計算
        eval_result = sum(accuracies) / len(accuracies)

        # モデル全体の評価のログの記録
        self.logger.log_global_eval(eval_result)

    # モデルの精度を評価するメソッド
    def calculate_fitness(self, training_sessions, picked_eval):
        self.logger.log_progress("Start calculate_fitness...")

        # モデル出力を作成
        outputs = self.generate_completion(training_sessions["model"], picked_eval)

        # モデル出力を評価
        evaluation_result = self.calculate_success_rate_from_exec(outputs, picked_eval)

        # モデル毎の評価のログの記録
        self.logger.log_model_eval(training_sessions, evaluation_result)

        self.logger.log_progress(f"Evaluation result: {evaluation_result}, Model ID: {training_sessions['id']}")

        return evaluation_result

    # 精度評価の対象のモデル出力を作成するメソッド
    def generate_completion(self, model, tasks):
        self.logger.log_progress("Start generate_completion...")

        # 評価指標の初期化
        model.eval()
        if config.use_cpu:
            model = model.to(self.accelerator.device)
            model = self.accelerator.prepare(model)

        answers_list = {}
        counter = 1
        for batch in tqdm(tasks):
            self.logger.log_progress(f"Evaluation: {counter} / {config.eval_sample_size}")

            # プロンプトを生成
            prompt_curr = self.generate_prompt(batch[config.eval_input_column], batch[config.eval_test_name_column][0])

            # バッチをデバイスに転送
            inputs = self.student_tokenizer(prompt_curr, return_tensors="pt", padding="max_length", truncation=True, max_length=config.input_max_length)
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            inputs = self.accelerator.prepare(inputs)

            # 文字列へデコード
            with self.accelerator.no_sync(model):
                outputs = model.generate(**inputs, max_new_tokens=config.output_max_new_tokens, do_sample=True, top_p=config.eval_top_p, top_k=config.eval_top_k, temperature=config.eval_temperature)
            completion = self.student_tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            answers_list[batch["task_id"]] = extract_python_code(completion)
            counter += 1

        if config.use_cpu:
            model.to("cpu")

        return answers_list

    # モデル出力の評価を行うメソッド
    def calculate_success_rate_from_exec(self, outputs, tasks):
        self.logger.log_progress("Start calculate_success_rate_from_exec...")

        results = {}
        timeout = 3.0
        for task in tqdm(tasks):
            try:
                # 実行する関数の名前を取得し、実行可能なコードを作成
                tests = "\n".join(task[config.eval_test_name_column])
                exec_code = outputs[task["task_id"]] + "\n" + tests
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        # 実行
                        exec(exec_code, exec_globals)

                # 結果を辞書に追加
                print("done")
                results[task["task_id"]] = True
            except Exception as e:
                print(e)
                results[task["task_id"]] = False

        # Trueの数をカウント
        num_true = sum(value for value in results.values())

        # 総数を取得
        total = len(results)

        # Trueの割合を計算
        return num_true / total

    # 評価時のプロンプトを生成するメソッド
    def generate_prompt(self, instruction: str, tests: str) -> str:
        return f'"""{instruction}\nTests: {tests}\nCode:"""'.strip()
