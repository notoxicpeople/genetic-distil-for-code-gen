# 各JSONエントリを処理
import json as json2
import re

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer

import config.config as config

output_max_length = 1024  # 質問のモデル出力の最大長
train_batch_size = 1  # 一回の訓練時のバッチサイズ
dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
student_tokenized_dataset_path = "input/tokens/CodeExercise-Python-27k+python_code_instructions_18k_alpaca/" + config.student_model_name
teacher_tokenized_dataset_path = "input/tokens/CodeExercise-Python-27k+python_code_instructions_18k_alpaca/" + config.teacher_model_name


def load_json(path):
    with open(path, "r") as f:
        return json2.load(f)


def process_json(input_json):
    processed_json = []

    for item in input_json:
        instruction = item["chat_rounds"][0]["content"]
        output = item["chat_rounds"][1]["content"]
        # コード部分を抽出
        code_match = re.search(r"```python\n(.*?)```", output, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            output = code.replace("python\n", "")
            output = re.sub(r"(\S)[ ]+#", r"\1#", output)
            output = re.sub(r"(\S)[ ]+\"\"\"", r"\1\"\"\"", output)
            output = re.sub(r"#.*?\n", "\n", output)
            output = re.sub(r"\n[ ]*\"\"\".*?\"\"\"\n", "\n", output, flags=re.DOTALL)
            output = re.sub(r"\n+", "\n", output)

        processed_json.append({"instruction": instruction, "output": output})

    # 結合されたテキストを作成
    texts = [f"{item['instruction']}\nCode:{item['output']}\n\n" for item in processed_json]
    return texts


def tokenize(texts, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "mps"

    # トークン化関数の定義
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=output_max_length, return_attention_mask=True).to(device)
        clm_labels = tokens["input_ids"].new(tokens["input_ids"].size()).copy_(tokens["input_ids"])
        clm_labels[~tokens["attention_mask"].bool()] = -100
        tokens["labels"] = clm_labels
        return tokens

    dataset = Dataset.from_dict({"text": texts})
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_datasets


def process_dataset():
    json_dataset = load_dataset(dataset_name)
    train = json_dataset["train"]
    combined_x = ["{instruction}\nInput:{input}\n" for x, alpha in zip(train["instruction"], train["input"])]
    combined_xy = [f"{x}\nCode:{y}\n\n" for x, y in zip(combined_x, train["output"])]
    return combined_xy


json = load_json(config.dataset_json_file_path)
json_texts = process_json(json)
dataset_texts = process_dataset()
device = "cuda" if torch.cuda.is_available() else "mps"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name, torch_dtype=torch_dtype, cache_dir=config.student_tokenizer_cache_path, trust_remote_code=True, padding_side="left")
student_tokenizer.pad_token = student_tokenizer.eos_token

teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name, torch_dtype=torch_dtype, cache_dir=config.teacher_tokenizer_cache_path, trust_remote_code=True, padding_side="left")
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_tokenized_json: Dataset = tokenize(json_texts, teacher_tokenizer)  # type: Dataset
student_tokenized_json: Dataset = tokenize(json_texts, student_tokenizer)  # type: Dataset
teacher_tokenized_dataset: Dataset = tokenize(dataset_texts, teacher_tokenizer)  # type: Dataset
student_tokenized_dataset: Dataset = tokenize(dataset_texts, student_tokenizer)  # type: Dataset

teacher_combined_datasets = ConcatDataset([teacher_tokenized_json, teacher_tokenized_dataset])
student_combined_datasets = ConcatDataset([student_tokenized_json, student_tokenized_dataset])

teacher_tokenized_dataset.save_to_disk(teacher_tokenized_dataset_path)
student_tokenized_dataset.save_to_disk(student_tokenized_dataset_path)
teacher_tokenized_json.save_to_disk(config.student_dataset_path)
student_tokenized_json.save_to_disk(config.teacher_dataset_path)


def count_total_tokens(dataset_path):
    # 保存されたデータセットをロード
    tokenized_dataset = load_from_disk(dataset_path)

    # トークン数を集計
    total_tokens = 0
    for sample in tokenized_dataset:
        # input_idsの長さを数えることでトークン数を得る
        token_count = len(sample["input_ids"])
        total_tokens += token_count

    return total_tokens


teacher_total_tokens = count_total_tokens(teacher_tokenized_dataset_path)
student_total_tokens = count_total_tokens(student_tokenized_dataset_path)

print(f"Teacher model total tokens: {teacher_total_tokens}")
print(f"Student model total tokens: {student_total_tokens}")
