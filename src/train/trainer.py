"""
Trainer class for KoUniPunc
"""
from typing import Literal
import os
import shutil
import logging
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

        self.model = KoUniPunc(args)
        self.model.to(self.device)

        self.test_texts = None
        if args.write_pred:
            self.test_texts = get_eval_texts(args)
            # Empty the original prediction files
            if os.path.exists(args.pred_dir):
                shutil.rmtree(args.pred_dir)

    def _init_trainer(self, total_data_len: int):
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

        return optimizer, scheduler

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        optimizer, scheduler = self._init_trainer(len(train_dataloader))

        # Train!
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                # logger.info(f"BATCH :: {batch}")

                inputs = {
                    "text_input_ids": batch[0],
                    "text_attention_mask": batch[1],
                    "text_token_type_ids": batch[2],
                    "labels": batch[3],
                    "text_length": batch[4],
                    "audio_input": batch[5],
                    "audio_length": batch[6],
                    "has_audio": batch[7][0],
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
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if (
                        self.args.logging_steps > 0
                        and global_step % self.args.logging_steps == 0
                    ):
                        self.evaluate("dev", global_step)

                    if (
                        self.args.save_steps > 0
                        and global_step % self.args.save_steps == 0
                    ):
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode: Literal["dev", "test"], step):
        dataset_map = {"dev": self.dev_dataset, "test": self.test_dataset}

        if mode not in ["dev", "test"]:
            raise Exception("Only dev and test dataset available")

        dataset = dataset_map[mode]
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "text_input_ids": batch[0],
                    "text_attention_mask": batch[1],
                    "text_token_type_ids": batch[2],
                    "labels": batch[3],
                    "text_length": batch[4],
                    "audio_input": batch[5],
                    "audio_length": batch[6],
                    "has_audio": batch[7][0],
                }
                outputs = self.model(**inputs)
                # outputs = self.model(**inputs).tolist()
                tmp_eval_loss, logits = outputs[:2]

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
            if not os.path.exists(self.args.pred_dir):
                os.mkdir(self.args.pred_dir)

            with open(
                os.path.join(self.args.pred_dir, f"pred_{mode}_{step}.txt"),
                "w",
                encoding="utf-8",
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
            if not os.path.exists(self.args.report_dir):
                os.makedirs(self.args.report_dir)

            report_path = os.path.join(self.args.report_dir, f"report_{step}.csv")
            df = pd.DataFrame(report).transpose()
            df.to_csv(report_path, sep=",")
            logger.info("Saved evaluated results!")

        else:
            logger.info("\n" + report)

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_ckpt_dir):
            os.makedirs(self.args.model_ckpt_dir)

        # model_to_save = (
        #     self.model.module if hasattr(self.model, "module") else self.model
        # )
        # model_to_save.save_pretrained(self.args.model_ckpt_dir)

        # save model
        torch.save(
            self.model.state_dict(),
            os.path.join(self.args.model_ckpt_dir, "kounipunc_state.pt"),
        )

        # Save training arguments together with the trained model
        torch.save(
            self.args, os.path.join(self.args.model_ckpt_dir, "training_args.bin")
        )
        logger.info("Saving model checkpoint to %s", self.args.model_ckpt_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_ckpt_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            # self.model = self.model_class.from_pretrained(self.args.model_ckpt_dir)
            state_dict = torch.load(
                os.path.join(self.args.model_ckpt_dir, "kounipunc_state.pt")
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")

        except:
            raise Exception("Some model files might be missing...")
