from datetime import datetime

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

import config.config as config

trained_model = f"output/model/by_trainer/{datetime.now().strftime('%Y%m%d-%H%M%S')}/{config.student_model_name}"
logs = f"output/log/by_trainer/{datetime.now().strftime('%Y%m%d-%H%M%S')}/{config.student_model_name}"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1": f1}


# モデルとトークナイザーのロード
tokenizer = AutoTokenizer.from_pretrained(config.student_model_name, cache_dir=config.student_tokenizer_cache_path, trust_remote_code=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(config.student_model_name, device_map="auto", cache_dir=config.student_model_cache_path)
tokenizer.pad_token = tokenizer.eos_token

# データセットのロードとトークナイズ
# tokenized_dataset = load_from_disk("output/tokens/CodeExercise-Python-27k+python_code_instructions_18k_alpaca/" + model_name)
tokenized_dataset = load_from_disk(config.student_dataset_path)
split_datasets = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

training_args = TrainingArguments(
    output_dir="output/results",  # 出力ディレクトリ
    num_train_epochs=1,  # トレーニングのエポック数
    per_device_train_batch_size=1,  # バッチサイズ
    per_device_eval_batch_size=1,  # 評価時のバッチサイズ
    gradient_accumulation_steps=4,
    warmup_steps=600,  # ウォームアップステップ数
    weight_decay=0.01,  # 重み減衰
    logging_dir=logs,  # ログのディレクトリ
    logging_steps=10,
    max_grad_norm=5.0,
    load_best_model_at_end=True,
    save_total_limit=1,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    save_strategy="steps",
    # fp16=True,  # ハーフプレシジョンを有効にする
)

# Trainerの作成
trainer = Trainer(
    compute_metrics=compute_metrics,
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# トレーニングの開始
trainer.train()
# trainer.evaluate()
trainer.save_model(trained_model)
