import argparse
import os

from model_running.dataset.custom_dataset import JointCapPuncDataset
from model_running.model_utils.model_executor import ModelExecutor


parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("--data-folder", type=str, required=True)
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--cache-dir", type=str, required=False)
parser.add_argument("--max-words", type=int, required=True)
parser.add_argument("--max-len", type=int, required=True)
parser.add_argument("--overlap", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-cap", type=int, required=True)
parser.add_argument("--num-punc", type=int, required=True)
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--num-epochs", type=int, required=True)


if __name__ == "__main__":
    args = parser.parse_args()

    train_data = JointCapPuncDataset(
        data_path=os.path.join(args.data_folder, "train.txt"),
        tokenizer_name=args.model_name,
        cache_dir=args.cache_dir,
        max_words=args.max_words,
        max_len=args.max_len,
        overlap=args.overlap
    )
    dev_data = JointCapPuncDataset(
        data_path=os.path.join(args.data_folder, "dev.txt"),
        tokenizer_name=args.model_name,
        cache_dir=args.cache_dir,
        max_words=args.max_words,
        max_len=args.max_len,
        overlap=args.overlap
    )
    test_data = JointCapPuncDataset(
        data_path=os.path.join(args.data_folder, "test.txt"),
        tokenizer_name=args.model_name,
        cache_dir=args.cache_dir,
        max_words=args.max_words,
        max_len=args.max_len,
        overlap=args.overlap
    )

    executor = ModelExecutor(
        batch_size=args.batch_size,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        num_cap=args.num_cap,
        num_punc=args.num_punc,
        cache_dir=args.cache_dir,
        num_epochs=args.num_epochs
    )

    executor.run(train_data, dev_data, test_data)
