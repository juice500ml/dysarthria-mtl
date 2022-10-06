import argparse
import itertools
import json
from pathlib import Path
from datetime import datetime
import pickle

from sklearn import metrics
import numpy as np
import tqdm
import pandas as pd
import transformers
from datasets import load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchaudio
from jiwer import wer
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Model,
)


def _prepare_cfg(raw_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the model. \n\n"
        ),
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="N/A"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="N/A"
    )
    parser.add_argument(
        "--num_classes", type=int, default=2, help="N/A"
    )
    parser.add_argument(
        "--ctc_weight", type=float, default=0.5, help="N/A"
    )
    parser.add_argument(
        "--csv_path", type=Path, default=Path("dataset.csv"),
    )
    parser.add_argument(
        "--target_metric", type=str, default="accuracy"
    )

    args = parser.parse_args(raw_args)  # Default to sys.argv
    args.exp_name = f"cls={args.num_classes}_e={args.num_epochs}_bs={args.batch_size}_ctcW={args.ctc_weight}"
 
    return args


class DysarthriaDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio, fs = torchaudio.load(row.path)
        audio = torchaudio.functional.resample(
            waveform=audio, orig_freq=fs, new_freq=16_000,
        )[0]

        audio_len = len(audio)

        cls_label = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        }[row.category]

        ctc_label = self.tokenizer.encode(row.text)
        
        return {
            "audio": audio,
            "audio_len": audio_len,
            "cls_label": cls_label,
            "ctc_label": ctc_label,
        }

    def __len__(self):
        return len(self.df)


def _collator(batch):
    return {
        "input_values": torch.nn.utils.rnn.pad_sequence(
            [x["audio"] for x in batch],
            batch_first=True,
            padding_value=0.0,
        ),
        "input_lengths": torch.LongTensor(
            [x["audio_len"] for x in batch]
        ),
        "cls_labels": torch.LongTensor(
            [x["cls_label"] for x in batch]
        ),
        "ctc_labels": torch.nn.utils.rnn.pad_sequence(
            [torch.IntTensor(x["ctc_label"]) for x in batch],
            batch_first=True,
            padding_value=-100,
        ),
    }


def get_tokenizer(root_dir, df):
    vocabs = set(itertools.chain.from_iterable(
        df.text.apply(lambda x: x.split())))
    vocab_dict = {v: i for i, v in enumerate(vocabs)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(root_dir / "vocab.json", "w") as f:
        json.dump(vocab_dict, f)

    return Wav2Vec2CTCTokenizer(
        root_dir / "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" ",
    )


def _get_dataset(tokenizer, target_df, **kwargs):
    return torch.utils.data.DataLoader(
        DysarthriaDataset(target_df, tokenizer),
        batch_size=1, collate_fn=_collator, pin_memory=True, **kwargs,
    )


def _prepare_dataset(root_dir, df):
    # Random split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["category"])
    train_df, valid_df = train_test_split(train_df, test_size=0.25, random_state=101, stratify=train_df["category"])

    train_df.to_csv(root_dir / "train.csv", index=False)
    valid_df.to_csv(root_dir / "valid.csv", index=False)
    test_df.to_csv(root_dir / "test.csv", index=False)

    tokenizer = get_tokenizer(root_dir, df)
    train_ds = _get_dataset(tokenizer, train_df, shuffle=True)
    valid_ds = _get_dataset(tokenizer, valid_df, shuffle=False)
    test_ds = _get_dataset(tokenizer, test_df, shuffle=False)

    return tokenizer, train_ds, valid_ds, test_ds


class Wav2Vec2MTL(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config.task_specific_params

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.cls_head = nn.Linear(config.hidden_size, self.cfg["num_classes"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self._vocab_size = config.vocab_size

    def forward(
        self,
        input_values,
        input_lengths,
        cls_labels,
        ctc_labels,
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=input_lengths[:, None],
            return_dict=False,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Classification
        _, max_state_len, _ = hidden_states.shape
        state_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        mask = (torch.arange(max_state_len)[None, :].to(state_lens.device) < state_lens[:, None])[:, :, None]
        avg_states = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)
        cls_logits = self.cls_head(avg_states)
        cls_loss = F.cross_entropy(cls_logits, cls_labels)

        # CTC
        labels_mask = ctc_labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = ctc_labels.masked_select(labels_mask)
        ctc_logits = self.lm_head(hidden_states)
        log_probs = nn.functional.log_softmax(ctc_logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                state_lens,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )

        # Final loss
        loss = cls_loss + self.cfg["ctc_weight"] * ctc_loss
        return (
            loss,
            cls_loss,
            ctc_loss,
            avg_states,
            hidden_states,
            cls_logits,
            ctc_logits,
        )

def _prepare_model(args_cfg, tokenizer):
    cfg, *_ = transformers.PretrainedConfig.get_config_dict("facebook/wav2vec2-xls-r-300m")
    cfg["gradient_checkpointing"] = True
    cfg["task_specific_params"] = {
        "num_classes": args_cfg.num_classes,
        "ctc_weight": args_cfg.ctc_weight,
    }
    cfg["vocab_size"] = len(tokenizer)
    return Wav2Vec2MTL.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        config=transformers.Wav2Vec2Config.from_dict(cfg),
    ).to(torch.device("cpu"))


def _eval(model, ds, tokenizer):
    def _ctc_decode(token_ids):
        # Output string with space in-between.
        return tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )

    loop = tqdm.tqdm(enumerate(ds))

    cls_all, ctc_all = [], []
    cls_labels, ctc_labels = [], []

    for step, x in loop:
        assert len(x["cls_labels"]) == 1

        cls_labels.append(x["cls_labels"].item())
        ctc_labels.append(_ctc_decode(x["ctc_labels"].numpy()[0]))

        x = {k: v.to(model.device) for k, v in x.items()}
        *_, cls_logits, ctc_logits = model(**x)

        cls_all.append(cls_logits.detach().numpy())
        pred_ids = np.argmax(ctc_logits.detach().numpy(), axis=-1)
        ctc_all.append(_ctc_decode(pred_ids))

    cls_all = np.concatenate(cls_all)
    prob_all = softmax(cls_all, axis=1)

    return {
        "accuracy": metrics.accuracy_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1)),
        "precision": metrics.precision_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "recall": metrics.recall_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "f1": metrics.f1_score(
            y_true=cls_labels, y_pred=prob_all.argmax(1), average="macro"),
        "per": wer(truth=ctc_labels, hypothesis=ctc_all),
    }


def _train(cfg, model, train_ds, valid_ds, tokenizer, ckpt_path, logger):
    grad_acc = cfg.batch_size
    eval_target = None
    steps = 0

    for epoch in range(cfg.num_epochs):
        train_loop = tqdm.tqdm(enumerate(train_ds))

        # Train
        for step, x in train_loop:
            x = {k: v.to(model.device) for k, v in x.items()}

            loss, cls_loss, ctc_loss, *_ = model(**x)
            (loss / grad_acc).backward()

            _losses = {"loss": loss.item(), "ctc_loss": ctc_loss.item(), "cls_loss": cls_loss.item()}
            train_loop.set_description(
                " | ".join([f"Epoch [{epoch}] "] + [f"{k} {v:.4f}" for k, v in _losses.items()])
            )
            for k, v in _losses.items():
                logger(f"train/{k}", v, steps)
            steps += 1

            # NOTE: gradient accumulation step == batch size in this code.
            if step % grad_acc == (grad_acc - 1):
                optimizer.step()
                optimizer.zero_grad()

        # Evaluation
        eval_results = _eval(model, valid_ds, tokenizer)
        for k, v in eval_results.items():
            logger(f"eval/{k}", v, epoch)

        # Bestkeeping
        if eval_target is None or eval_target <= eval_results[cfg.target_metric]:
            if eval_target is None:
                eval_target = 0.0
            print(
                f"Updating the model with better {cfg.target_metric}.\n"
                f"Prev: {eval_target:.4f}, Curr (epoch={epoch}): {eval_results[cfg.target_metric]:.4f}\n"
                f"Removing the previous checkpoint.\n"
            )
            eval_target = eval_results[cfg.target_metric]
            model.save_pretrained(ckpt_path)


def _get_logger(tb_path):
    writer = SummaryWriter(log_dir=tb_path)
    def _log(name, value, step=0):
        writer.add_scalar(name, value, step)
    return _log


if __name__ == "__main__":
    cfg = _prepare_cfg()
    print(cfg)

    root_dir = Path(f'exp_results/{datetime.today().strftime("%Y-%m-%d_%H:%M:%S")}_{cfg.exp_name}')
    root_dir.mkdir(parents=True, exist_ok=False)

    pickle.dump(cfg, open(root_dir / "experiment_args.pkl", "wb"))
    ckpt_path = root_dir / "best-model-ckpt"
    logger = _get_logger(root_dir)

    tokenizer, train_ds, valid_ds, test_ds = _prepare_dataset(root_dir, pd.read_csv(cfg.csv_path))

    model = _prepare_model(cfg, tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9,0.98), eps=1e-08)

    # Train & Validation loop
    _train(cfg, model, train_ds, valid_ds, tokenizer, ckpt_path=ckpt_path, logger=logger)

    # Test on the best model
    best_model = Wav2Vec2MTL.from_pretrained(ckpt_path).to(torch.device("cpu"))
    test_results = _eval(best_model, test_ds, tokenizer,processor)
    print(test_results)
    for k, v in test_results.items():
        logger(f"test/{k}", v, 0)
    json.dump(test_results, open(root_dir / "test_metric_results.json", "w"))
