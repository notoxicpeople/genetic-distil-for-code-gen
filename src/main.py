# sudo apt update
# sudo apt install python3-pip
# pip install bitsandbytes accelerate datasets transformers torch tqdm einops

from training_process import TrainingProcess

if __name__ == "__main__":
    process = TrainingProcess()
    process.main_process()
