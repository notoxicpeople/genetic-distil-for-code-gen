import os

from accelerate import Accelerator
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config.config as config
from sigleton.training_process_singleton import TrainingProcessSingleton
from util.code_extract import extract_python_code
from sigleton.logger_singleton import LoggerSingleton


class ModelEval:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.logger: LoggerSingleton = LoggerSingleton()
        self.accelerator: Accelerator = singleton.accelerator
        self.now = singleton.now
        self.student_tokenizer: AutoTokenizer = singleton.student_tokenizer
        self.training_sessions: list = singleton.training_sessions

    def save_output_for_eval(self):
        for i, training_session in enumerate(self.training_sessions):
            rank = i + 1
            id = training_session["id"]
            self.generate_completions_and_save(training_session, f"end_process/{self.now}/{rank}_{id}")

    def generate_completion(self, prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(f"{prompt}", return_tensors="pt", padding=True, truncation=True, max_length=config.input_max_length)
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        inputs = self.accelerator.prepare(inputs)

        with self.accelerator.no_sync(model):
            generate_ids = model.generate(**inputs, max_new_tokens=config.output_max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95)
        completion = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        completion = extract_python_code(completion)

        return completion

    def generate_completions_and_save(self, training_session, model_path):
        model = training_session["model"]
        model.eval()
        if config.use_cpu:
            model = model.to(self.accelerator.device)
            model = self.accelerator.prepare(model)

        output_path = f"{config.eval_path}/{model_path}/{self.now}"
        problems = read_problems()
        num_samples_per_task = config.num_samples_per_task_for_humaneval
        samples = [
            dict(task_id=task_id, completion=self.generate_completion(problems[task_id]["prompt"], model, self.student_tokenizer)) for task_id in tqdm(problems) for _ in range(num_samples_per_task)
        ]
        os.makedirs(output_path, exist_ok=True)
        write_jsonl(f"{output_path}/samples.jsonl", samples)

        if config.use_cpu:
            model.to("cpu")
