import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader, RandomSampler, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

import config.config as config
from sigleton.training_process_singleton import TrainingProcessSingleton
from config.config import TeachMode
from sigleton.logger_singleton import LoggerSingleton


class ModelTrain:
    def __init__(self):
        singleton: TrainingProcessSingleton = TrainingProcessSingleton()
        self.training_sessions: list = singleton.training_sessions
        self.accelerator: Accelerator = singleton.accelerator
        self.logger: LoggerSingleton = LoggerSingleton()

        # 教師モデルのロード
        if config.teach_mode == TeachMode.BY_TEACHER:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                config.teacher_model_name, torch_dtype=singleton.torch_dtype, trust_remote_code=True, device_map="auto", cache_dir=config.teacher_model_cache_path
            )
            self.teacher_model = self.accelerator.prepare(self.teacher_model)

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name, torch_dtype=singleton.torch_dtype, cache_dir=config.teacher_tokenizer_cache_path, trust_remote_code=True, padding_side="left"
        )
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        self.teacher_tokenized_dataset = load_from_disk(config.teacher_dataset_path)
        self.student_tokenized_dataset = load_from_disk(config.student_dataset_path)
        self.teacher_data_collator = DataCollatorForLanguageModeling(tokenizer=teacher_tokenizer, mlm=False)
        self.student_data_collator = DataCollatorForLanguageModeling(tokenizer=singleton.student_tokenizer, mlm=False)

    def train_models(self):
        # 教師モデル（最も精度の高いモデル）とcalculate_fitness生徒モデル（新たに追加されたモデル）の訓練
        loss = 0.0
        for training_session in self.training_sessions:
            # 生徒モデルの訓練
            loss += self.train(training_session)
        return loss

    # 教師モデルと生徒モデルの訓練を行うメソッド
    def train(self, training_session):
        self.logger.log_progress("Start train...")

        # 精度の高いモデルを格納
        self.top_training_session = self.training_sessions[0]

        # 生徒モデルの取得
        student_model = training_session["model"]
        student_optimizer = training_session["optimizer"]
        distillation_weight = training_session["distillation_weight"]
        classification_weight = training_session["classification_weight"]
        training_session["accuracy"] = 0

        # 学習率のスケジューラーの初期化
        scheduler = get_linear_schedule_with_warmup(
            student_optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.num_train_optimization_steps,
        )

        student_model.train()
        if config.use_cpu:
            student_model = student_model.to(self.accelerator.device)
            student_model = self.accelerator.prepare(student_model)
        is_top_student = self.top_training_session["model"] == student_model

        if config.teach_mode == TeachMode.BY_TEACHER:
            self.teacher_model.eval()
        elif config.teach_mode == TeachMode.BY_TOP_STUDENT and is_top_student is False:
            self.top_training_session["model"].eval()
            if config.use_cpu:
                self.top_training_session["model"] = self.top_training_session["model"].to(self.accelerator.device)
                self.top_training_session["model"] = self.accelerator.prepare(self.top_training_session["model"])

        # データローダーの取得
        teacher_dataloader, student_dataloader = self.get_shuffled_dataloaders()

        # バッチ処理
        accumulated_loss = 0.0  # 蓄積された損失（記録用）
        total_loss = 0.0  # 訓練を通しての損失（早期終了用）
        total_accuracy = 0.0  # 訓練を通しての精度

        for epoch in range(config.train_epochs):
            self.logger.log_progress(f"Train iteration: {epoch + 1} / {config.train_epochs}")
            for i, (student_batch, teacher_batch) in enumerate(tqdm(zip(student_dataloader, teacher_dataloader), total=config.train_batches_per_epoch)):
                # 指定したバッチ数に達したらループを終了
                if i > config.train_batches_per_epoch - 1:
                    break

                # 損失の取得
                accuracy, loss_distillation, loss_classification = self.step_clm(teacher_batch, student_batch, student_model, distillation_weight, classification_weight)
                loss = loss_distillation + loss_classification
                total_accuracy += accuracy

                # バックプロパゲーション
                total_loss += loss.item()  # 訓練を通しての損失の更新
                loss = loss / config.accumulation_steps  # 勾配累積に対応
                accumulated_loss += loss.item() * config.accumulation_steps  # 累積損失の更新
                self.accelerator.backward(loss)

                # ログの記録
                self.logger.log_train(training_session, scheduler, loss, loss_distillation, loss_classification, accuracy)

                # 勾配を指定されたステップ数だけ蓄積した後に更新
                if (i + 1) % config.accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
                    student_optimizer.step()
                    scheduler.step()
                    student_optimizer.zero_grad()
                    self.loss_history.append(accumulated_loss / config.accumulation_steps)
                    accumulated_loss = 0.0

                self.logger.log_progress(
                    f"Train {i+1}/{config.train_batches_per_epoch}, Loss: {loss.item()}, Distillation Loss: {loss_distillation.item()}, Classification Loss: {loss_classification.item()}, Accuracy: {accuracy}"
                )

            # 最後のバッチで蓄積された勾配が残っている場合、それも更新する
            if config.train_batches_per_epoch % config.accumulation_steps != 0:
                student_optimizer.step()
                student_optimizer.zero_grad()

            # 訓練の終了処理
            training_session["accuracy"] = total_accuracy / config.train_batches_per_epoch
            self.logger.log_train_epoch(train_session=training_session)

        if config.use_cpu:
            student_model.to("cpu")
            if config.teach_mode == TeachMode.BY_TOP_STUDENT and is_top_student is False:
                self.top_training_session["model"].to("cpu")

        return total_loss

    # clmによる損失の計算
    def step_clm(self, teacher_batch, student_batch, student_model, distillation_weight, classification_weight):
        teacher_input_ids = teacher_batch["input_ids"]
        teacher_attention_mask = teacher_batch["attention_mask"]
        teacher_lm_labels = teacher_batch["labels"]
        student_input_ids = student_batch["input_ids"]
        student_attention_mask = student_batch["attention_mask"]
        student_lm_labels = student_batch["labels"]
        is_top_student = self.top_training_session["model"] == student_model
        student_outputs = student_model(input_ids=student_input_ids, attention_mask=None)

        # 教師モデルが存在する場合、または生徒モデルが最も精度の高いモデルでない場合に蒸留損失を計算
        # 生徒モデルが最も精度の高いモデル場合、自身が教師モデルとなるため。
        loss_distillation = torch.tensor(0.0)
        device = next(student_model.parameters()).device
        loss_distillation = loss_distillation.to(device)
        if config.teach_mode == TeachMode.BY_TEACHER:
            loss_distillation = self.calculate_distillation_loss(self.teacher_model, teacher_input_ids, student_outputs, student_attention_mask, distillation_weight)
        elif config.teach_mode == TeachMode.BY_TOP_STUDENT and is_top_student is False:
            loss_distillation = self.calculate_distillation_loss(self.top_training_session["model"], student_input_ids, student_outputs, student_attention_mask, distillation_weight)
        elif config.teach_mode == TeachMode.BY_EVERY_STUDENTS:
            for training_session in self.training_sessions:
                if training_session["model"] == student_model:
                    continue
                training_session["model"].eval()
                training_session["model"] = training_session["model"].to(self.accelerator.device)
                training_session["model"] = self.accelerator.prepare(training_session["model"])
                loss_distillation += self.calculate_distillation_loss(training_session["model"], student_input_ids, student_outputs, student_attention_mask, distillation_weight)
                if config.use_cpu:
                    training_session["model"].to("cpu")

        # 分類損失の計算
        s_logits = student_outputs["logits"]
        shift_logits = s_logits[..., :-1, :].contiguous()
        shift_labels = student_lm_labels[..., 1:].contiguous()
        loss_clm = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_classification = classification_weight * loss_clm

        # 正確性の計算
        accuracy_top1, accuracy_top5 = 0, 0
        if config.eval_by_exec is False:
            accuracy_top1, accuracy_top5 = self.accuracy_topk(shift_logits, shift_labels, topk=(1, 5))

        return accuracy_top5, loss_distillation, loss_classification

    def calculate_distillation_loss(self, teacher_model, teacher_input_ids, student_outputs, student_attention_mask, distillation_weight):
        """
        教師モデルと生徒モデルの出力を用いて蒸留損失を計算する。

        :param teacher_outputs: 教師モデルの出力
        :param student_outputs: 生徒モデルの出力
        :param student_attention_mask: 生徒モデルのアテンションマスク
        :param config.distillation_temperature: 蒸留時の温度パラメータ
        :param distillation_weight: 蒸留損失の重み
        :return: 蒸留損失
        """

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=teacher_input_ids, attention_mask=None)

        t_logits = teacher_outputs["logits"]
        s_logits = student_outputs["logits"]
        assert s_logits.size() == t_logits.size()

        student_mask = student_attention_mask.unsqueeze(-1).expand_as(s_logits)
        student_mask_bool = student_mask.to(torch.bool)

        # 教師モデルの出力から、生徒モデルの出力に対応する部分を抽出
        s_logits_slct = torch.masked_select(s_logits, student_mask_bool)
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))
        # マスクに生徒モデルのマスクを使用する。
        # マスクが異なると、t_logits_slctとs_logits_slctのサイズが合わなくなる。
        t_logits_slct = torch.masked_select(t_logits, student_mask_bool)
        t_logits_slct = t_logits_slct.view(-1, t_logits.size(-1))
        assert t_logits_slct.size() == s_logits_slct.size()

        # 蒸留損失の計算
        loss_ce = (
            nn.KLDivLoss(reduction="batchmean")(
                nn.functional.log_softmax(s_logits_slct / config.distillation_temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / config.distillation_temperature, dim=-1),
            )
            * (config.distillation_temperature) ** 2
        )
        loss_distillation = distillation_weight * loss_ce

        return loss_distillation

    # 正確性の計算
    def accuracy_topk(self, output, target, topk=(1,)):
        # 出力の寸法とデータ型を確認し、必要に応じて調整
        output = output.reshape(-1, output.size(-1))  # viewからreshapeに変更
        target = target.reshape(-1).long()  # viewからreshapeに変更

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.expand_as(pred))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)  # viewからreshapeに変更
            accuracy_k = correct_k.mul_(100.0 / batch_size)
            accuracies.append(accuracy_k.item())

        return accuracies

    # データローダー内のバッチをシャッフルするメソッド
    def get_shuffled_dataloaders(self):
        # データセットのインデックス用にランダムなインデックスを生成
        indices = torch.randperm(len(self.teacher_tokenized_dataset)).tolist()

        # データセットのサブセットを作成
        teacher_subset = Subset(self.teacher_tokenized_dataset, indices)
        student_subset = Subset(self.student_tokenized_dataset, indices)

        # データローダーの作成
        teacher_dataloader = DataLoader(teacher_subset, batch_size=config.train_batch_size, collate_fn=self.teacher_data_collator, sampler=RandomSampler(teacher_subset))
        student_dataloader = DataLoader(student_subset, batch_size=config.train_batch_size, collate_fn=self.student_data_collator, sampler=RandomSampler(student_subset))

        teacher_dataloader = self.accelerator.prepare_data_loader(teacher_dataloader)
        student_dataloader = self.accelerator.prepare_data_loader(student_dataloader)

        return teacher_dataloader, student_dataloader
