from transformers import AutoModelForCausalLM, AutoTokenizer

import config.config as config

model = AutoModelForCausalLM.from_pretrained(config.student_model_name, trust_remote_code=True, device_map="auto", cache_dir=config.student_model_cache_path)
tokenizer = AutoTokenizer.from_pretrained(config.student_model_name, cache_dir=config.student_tokenizer_cache_path)
teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name, cache_dir=config.teacher_tokenizer_cache_path)

for name, param in model.named_parameters():
    print(name, param.size())


print(teacher_tokenizer.special_tokens_map)
