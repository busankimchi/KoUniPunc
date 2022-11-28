"""
Trainer class for KoUniPunc
"""
from typing import Literal
import os
import shutil
import logging
from pathlib import Path
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from ..utils import (
    PUNCTUATION_LABELS,
    get_device,
    compute_metrics,
    show_report,
    get_eval_texts,
)

from ..model.ko_unipunc import KoUniPunc

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.device = get_device(args)
        self.label_lst = PUNCTUATION_LABELS

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.test_texts = None
        if args.write_pred:
            self.test_texts = get_eval_texts(args)
            # Empty the original prediction files
            if os.path.exists(args.pred_dir):
                shutil.rmtree(args.pred_dir)

    def _init_trainer(self, total_data_len: int):
        loaded_res = self.load_model()

        resume = loaded_res["resume"] if loaded_res is not None else None

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (total_data_len // self.args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                total_data_len
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # optimizer and schedule (linear warmup and decay)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )

        if loaded_res is not None:
            optimizer.load_state_dict(loaded_res["optimizer"])

        # TODO: learning rate scheduler with NOAM, warm-up step 8000
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        return optimizer, scheduler, resume

    def _get_data_loader(self, mode: Literal["train", "dev"]):
        if mode == "train":
            train_sampler = RandomSampler(self.train_dataset)
            train_dataloader = DataLoader(
                self.train_dataset,
                sampler=train_sampler,
                batch_size=self.args.train_batch_size,
            )
            return train_dataloader

        elif mode == "dev":
            eval_sampler = SequentialSampler(self.dev_dataset)
            eval_dataloader = DataLoader(
                self.dev_dataset,
                sampler=eval_sampler,
                batch_size=self.args.eval_batch_size,
            )
            return eval_dataloader

        else:
            raise Exception("Only dev and test dataset available")

    def train(self):
        train_dataloader = self._get_data_loader("train")
        optimizer, scheduler, resume = self._init_trainer(len(train_dataloader))

        # Train!
        tr_loss, global_step = 0.0, 0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for epoch in train_iterator:
            if resume is not None and resume["epoch"] > epoch:
                continue

            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # logger.info(f"BATCH :: {batch[3]}")

                inputs = {
                    "text_input_ids": batch[0],
                    "text_attention_mask": batch[1],
                    "text_token_type_ids": batch[2],
                    "audio_input": batch[3],
                    "audio_length": batch[4],
                    "has_audio": batch[5][0],
                    "labels": batch[6],
                }
                outputs = self.model(**inputs)
                loss: Tensor = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_grad_norm
                    )

                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.args.logging_steps > 0
                        and global_step % self.args.logging_steps == 0
                    ):
                        self.evaluate(global_step)

                    if (
                        self.args.save_steps > 0
                        and global_step % self.args.save_steps == 0
                    ):
                        self.save_model(step, epoch, optimizer)

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, step):
        eval_dataloader = self._get_data_loader("dev")

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(self.dev_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "text_input_ids": batch[0],
                    "text_attention_mask": batch[1],
                    "text_token_type_ids": batch[2],
                    "audio_input": batch[3],
                    "audio_length": batch[4],
                    "has_audio": batch[5][0],
                    "labels": batch[6],
                }
                tmp_eval_loss, logits = self.model(**inputs)

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs["labels"].detach().cpu().numpy(),
                    axis=0,
                )

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.args.write_pred:
            save_dir = os.path.join(self.args.pred_dir, self.args.log_prefix)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            with open(
                os.path.join(save_dir, f"pred_{step}.txt"), "w", encoding="utf-8"
            ) as f:
                for text, true_label, pred_label in zip(
                    self.test_texts, out_label_list, preds_list
                ):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write(f"{t} {tl} {pl}\n")
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        # Get the report for each tag result
        report = show_report(out_label_list, preds_list, self.args.report_as_file)

        if self.args.report_as_file:
            save_dir = os.path.join(self.args.report_dir, self.args.log_prefix)
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            report_path = os.path.join(save_dir, f"report_{step}.csv")
            df = pd.DataFrame(report).transpose()
            df.to_csv(report_path, sep=",")
            logger.info("Saved evaluated results!")

        else:
            logger.info("\n" + report)

        return results

    def save_model(self, step, epoch, optimizer):
        # Save model checkpoint
        save_dir = os.path.join(self.args.model_ckpt_dir, self.args.log_prefix)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # save model
        torch.save(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(save_dir, f"kounipunc_{epoch}.pt"),
        )

        # Save training arguments together with the trained model
        torch.save(
            self.args,
            os.path.join(save_dir, "kounipunc_args.bin"),
        )
        logger.info("Saving model checkpoint to %s", self.args.model_ckpt_dir)

    def load_model(self):
        self.model = KoUniPunc(self.args)
        self.model.to(self.device)

        if self.args.load_model_path is not None:
            # Check whether model exists
            if not os.path.exists(self.args.load_model_path):
                raise Exception("Model doesn't exists! Train first!")

            try:
                model_pt = torch.load(self.args.load_model_path)

                self.model.load_state_dict(model_pt["model_state_dict"])
                # self.model.to(self.device)
                logger.info("***** Model Loaded *****")

                return {
                    "optimizer": model_pt["optimizer_state_dict"],
                    "resume": {"epoch": model_pt["epoch"], "step": model_pt["step"]},
                }

            except:
                raise Exception("Some model files might be missing...")

        # self.model.to(self.device)
        logger.info("***** Model Loaded *****")

        return None
