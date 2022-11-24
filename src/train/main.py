"""
Training Entry file
"""
import argparse

from .trainer import Trainer
from ..utils import init_logger, set_seed
from ..dataset.data_loader import load_and_cache_examples
from ..utils import (
    LM_MODEL_CLASSES,
    LM_MODEL_PATH_MAP,
    SM_MODEL_CLASSES,
    SM_MODEL_PATH_MAP,
)


def main(args):
    init_logger()
    set_seed(args)

    train_dataset = load_and_cache_examples(args, mode="train")
    # train_dataset = load_and_cache_examples(args, mode="dev")
    dev_dataset = load_and_cache_examples(args, mode="dev")
    # test_dataset = load_and_cache_examples(args, mode="test")
    test_dataset = None

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """Dataset"""
    parser.add_argument(
        "--task", default="wfc-ko-punc", type=str, help="The name of the task to train"
    )

    """Trainer"""
    parser.add_argument(
        "--train_batch_size", default=2, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--eval_batch_size", default=4, type=int, help="Batch size for evaluation"
    )

    """KoUniPunc"""
    parser.add_argument(
        "--lm_model_type",
        type=str,
        default="kobert",
        help="Model type selected in the list: " + ", ".join(LM_MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--sm_model_type",
        default="wav2vec2_large_korean",
        type=str,
        help="Model type selected in the list: " + ", ".join(SM_MODEL_CLASSES.keys()),
    )
    parser.add_argument("--w2v_dim", type=int, default=1024, help="The w2v dim")

    parser.add_argument("--dropout", type=float, default=0.1, help="The dropout rate")
    parser.add_argument(
        "--ignore_wav",
        default=False,
        help="whether to ignore Wav, if set True, the model deduce to standard Transformer",
        action="store_true",
    )

    parser.add_argument(
        "--freeze_w2v", default=False, action="store_true", help="Freeze wav2vec module"
    )
    parser.add_argument(
        "--grad_scale_wav", type=int, default=None, help="Grad scale wav"
    )
    parser.add_argument(
        "--wav_mask_prob",
        type=float,
        default=0.0,
        help="The probability of the mask to wav",
    )

    parser.add_argument(
        "--max_seq_len", default=50, type=int, help="Max sentence length"
    )

    parser.add_argument(
        "--max_aud_len", default=480000, type=int, help="Max audio length"
    )

    parser.add_argument(
        "--wav_sampling_rate", type=int, default=16000, help="Audio data sampling rate"
    )

    """Header"""
    parser.add_argument(
        "--head_dropout", type=float, default=0.3, help="The dropout rate of header"
    )
    parser.add_argument(
        "--head_hidden_dim", type=int, default=768, help="hidden dimension of header"
    )
    parser.add_argument(
        "--head_cross_layer_number",
        type=int,
        default=2,
        help="The layer number of cross header",
    )
    parser.add_argument(
        "--head_layer_number", type=int, default=5, help="The layer number of header"
    )
    parser.add_argument(
        "--head_num", type=int, default=8, help="Number of heads per header"
    )

    """Virtual Embedding"""
    parser.add_argument(
        "--use_virtual",
        type=bool,
        default=True,
        help="Whether to use virtual embedding",
    )
    parser.add_argument(
        "--virtual_embed_dim",
        type=int,
        default=5,
        help="Dimension of virtual embedding",
    )

    """Conv1dSubsampler"""
    parser.add_argument(
        "--conv_channels",
        type=int,
        default=768,
        help="Dimension of conv channel for Conv1dSubsampler",
    )
    parser.add_argument(
        "--encoder_embed_dim",
        type=int,
        default=768,
        help="Dimension of output channel for Conv1dSubsampler",
    )
    parser.add_argument(
        "--conv_kernel_sizes",
        type=str,
        default="20,15",
        help="Conv kernel size for Conv1dSubsampler",
    )

    """Paths"""
    parser.add_argument(
        "--data_dir",
        default="/mnt/data_storage/sample_data",
        type=str,
        help="The input data dir",
    )
    parser.add_argument(
        "--model_dir", default="./model", type=str, help="Path for saving model"
    )
    parser.add_argument(
        "--pred_dir", default="./preds", type=str, help="The prediction file dir"
    )

    parser.add_argument(
        "--train_file", default="train.jsonl", type=str, help="Train file"
    )
    parser.add_argument("--dev_file", default="dev.jsonl", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test.jsonl", type=str, help="Test file")

    parser.add_argument(
        "--report_as_file",
        action="store_true",
        help="Whether to save prediction report as files. If false, print as stdout.",
    )
    parser.add_argument("--report_dir", default="./report", type=str, help="Report dir")

    """Training Parameters"""
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--learning_rate", default=0.00001, type=float, help="The initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=15.0,
        type=float,
        help="Total number of training epochs to perform.",
    )

    """Training General Options"""
    parser.add_argument(
        "--logging_steps", type=int, default=1200, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1200,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--write_pred", action="store_true", help="Write prediction during evaluation"
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the test set."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    args = parser.parse_args()
    args.lm_model_name_or_path = LM_MODEL_PATH_MAP[args.lm_model_type]
    args.sm_model_name_or_path = SM_MODEL_PATH_MAP[args.sm_model_type]

    main(args)
