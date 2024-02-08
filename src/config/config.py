from enum import Enum


class TeachMode(Enum):
    BY_TEACHER = 1
    BY_TOP_STUDENT = 2
    BY_EVERY_STUDENTS = 3


class Algorithm(Enum):
    DEBUG = 1
    GENETIC_ALGORITHM_UNIFORM = 2
    GENETIC_ALGORITHM_ARITHMETIC = 3


student_model_name = "Salesforce/codegen-350M-mono"
teacher_model_name = "microsoft/phi-2"
eval_dataset_name = "mbpp"
eval_input_column = "prompt"  # 評価データのプロンプト名
eval_test_name_column = "test_list"  # 評価データのテスト名
# dataset_name = "CodeExercise-Python-27k+python_code_instructions_18k_alpaca"
dataset_name = "CodeExercise-Python-27k"
student_tokenizer_cache_path = "cache/tokenizer/" + student_model_name
teacher_tokenizer_cache_path = "cache/tokenizer/" + teacher_model_name
student_model_cache_path = "cache/model/" + student_model_name
teacher_model_cache_path = "cache/model/" + teacher_model_name
student_dataset_path = f"input/tokens/{dataset_name}/{student_model_name}"
teacher_dataset_path = f"input/tokens/{dataset_name}/{teacher_model_name}"
dataset_json_file_path = "input/dataset/CodeExercise-Python-27k.json"
eval_path = "output/eval"
log_path = "output/log"
save_model_in_process_path = "output/model/trained/in_process"
save_model_at_end_path = "output/model/trained/end_process"

use_cpu = False  # CPUを使用するかどうか
use_early_stop = True  # 早期終了を使用するかどうか
eval_by_exec = True  # 実行による評価を使用するかどうか
teach_mode = TeachMode.BY_TEACHER  # 生徒モデルによる教師あり学習を使用するかどうか
algorithm = Algorithm.GENETIC_ALGORITHM_UNIFORM  # アルゴリズムの種類

total_iterations = 100  # 全体のプロセスの繰り返し回数
total_models = 3  # モデルの総数
eval_sample_size = 20  # 一回の精度を評価する際に使用するデータの数
output_max_new_tokens = 392  # 回答に関するモデル出力の最大長
input_max_length = 1024  # 質問の最大長
models_to_replace = 1  # 入れ替えるモデルの総数
train_epochs = 1  # 訓練のエポック数
train_batches_per_epoch = 512  # 訓練に使用するバッチ数
train_batch_size = 1  # 一回の訓練時のバッチサイズ
distillation_temperature = 2.0  # 蒸留時の温度
accumulation_steps = 8  # 勾配累積のステップ数
max_learning_rate = 1e-3  # 最大学習率
min_learning_rate = 1e-5  # 最小学習率
mutation_rate = 0.05  # 突然変異の確率
mutation_rate_for_hyper_param = 0.05  # 突然変異の確率
eval_top_p = 0.75  # 評価時のトークンサンプリングの確率
eval_top_k = 40  # 評価時のトークンサンプリングの確率
eval_temperature = 0.2  # 評価時の温度
max_grad_norm = 5.0  # 勾配クリッピングの閾値
log_interval = 16  # ログを記録する間隔
warmup_steps = train_batches_per_epoch * train_epochs * 0.1  # 学習率のスケジューラーのウォームアップステップ数
num_train_optimization_steps = train_batches_per_epoch * train_epochs  # 学習率のスケジューラーの総ステップ数
early_stop_count = 3  # 早期終了の閾値回数
early_stop_loss_change_ratio = 0.05  # 早期終了の損失変化の閾値
alpha_cross_over_hyper_parameter = 0.75  # 範囲拡張のためのパラメータ
num_samples_per_task_for_humaneval = 10  # HumanEvalの一つのタスクに対するサンプル数
