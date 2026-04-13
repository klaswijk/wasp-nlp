import matplotlib.pyplot as plt
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from datasets import load_dataset

from torch.utils.data import DataLoader, Subset
import numpy as np
import argparse
import tqdm
import sys, time, os


###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]


def build_tokenizer(
    train_file,
    tokenize_fun=lowercase_tokenizer,
    max_voc_size=None,
    model_max_length=None,
    pad_token="<PAD>",
    unk_token="<UNK>",
    bos_token="<BOS>",
    eos_token="<EOS>",
):
    """Build a tokenizer from the given file.

    Args:
         train_file:        The name of the file containing the training texts.
         tokenize_fun:      The function that maps a text to a list of string tokens.
         max_voc_size:      The maximally allowed size of the vocabulary.
         model_max_length:  Truncate texts longer than this length.
         pad_token:         The dummy string corresponding to padding.
         unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
         bos_token:         The dummy string corresponding to the beginning of the text.
         eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).

    # Loop through the training file, tokenize the lines, and update the counter.
    counter = Counter()
    with open(train_file, "r") as file:
        for line in file:
            # Skip empty lines.
            if line.strip() == "":
                continue

            # Tokenize the line and update the counter.
            tokens = tokenize_fun(line)
            counter.update(tokens)

    # Build the string-to-integer and integer-to-string mappings.
    # Include special tokens as the first items in the vocabulary,
    # then add the most common tokens from the counter until max_voc_size.
    str_to_int = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
    int_to_str = {0: pad_token, 1: unk_token, 2: bos_token, 3: eos_token}
    for i, (word, _) in enumerate(
        counter.most_common(max_voc_size), start=len(str_to_int)
    ):
        str_to_int[word] = i
        int_to_str[i] = word

    return A1Tokenizer(str_to_int, int_to_str, tokenize_fun, model_max_length)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, tokenize_fun, model_max_length):
        # TODO: store all values you need in order to implement __call__ below.
        self.pad_token_id = str_to_int.get("<PAD>")  # Compulsory attribute.
        self.unk_token_id = str_to_int.get("<UNK>")
        self.bos_token_id = str_to_int.get("<BOS>")
        self.eos_token_id = str_to_int.get("<EOS>")
        self.model_max_length = model_max_length  # Needed for truncation.
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.

        Args:
          texts:           The texts to tokenize.
          truncation:      Whether the texts should be truncated to model_max_length.
          padding:         Whether the tokenized texts should be padded on the right side.
          return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

        Returns:
          A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        if return_tensors and return_tensors != "pt":
            raise ValueError("Should be pt")

        # TODO: Your work here is to split the texts into words and map them to integer values.
        #
        # - If `truncation` is set to True, the length of the encoded sequences should be
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        max_length = 0
        input_ids = []
        for text in texts:
            tokens = self.tokenize_fun(text)
            ids = [self.str_to_int.get(token, self.unk_token_id) for token in tokens]

            if truncation and self.model_max_length is not None:
                ids = ids[: self.model_max_length - 2]

            max_length = max(max_length, len(ids) + 2)
            input_ids.append([self.bos_token_id] + ids + [self.eos_token_id])

        if padding:
            for ids in input_ids:
                ids += [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        return BatchEncoding({"input_ids": input_ids})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)

    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, "rb") as f:
            return pickle.load(f)


###
### Part 3. Defining the model.
###


class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""

    def __init__(self, vocab_size, embedding_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size


class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""

    config_class = A1RNNModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.RNN(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.

        Args:
          - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
          - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
        Returns:
          A CausalLMOutput containing
            - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
            - loss:     The loss computed on this batch.
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        if labels is not None:
            loss = self.loss_func(logits, labels)

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size:
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.


class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.

        Args:
          model:          The model to train.
          args:           The training parameters stored in a TrainingArguments object.
          train_dataset:  The dataset containing the training documents.
          eval_dataset:   The dataset containing the validation documents.
          eval_dataset:   The dataset containing the validation documents.
          tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert args.optim == "adamw_torch"
        assert args.eval_strategy == "epoch"

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device("cpu")
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mps.is_available():  # type: ignore
            return torch.device("mps")
        return torch.device("cpu")

    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print("Device:", device)
        self.model.to(device)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        if args.debug:
            self.train_dataset = Subset(self.train_dataset, range(1000))
            self.eval_dataset = Subset(self.eval_dataset, range(100))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        # TODO: Your work here is to implement the training loop.
        #
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
        #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()

        train_losses = []
        val_losses = []
        pbar = tqdm.trange(args.num_train_epochs, desc="Epoch")
        for _ in pbar:
            self.model.train()
            running_loss = 0.0
            for batch in tqdm.tqdm(train_loader, desc="Batch", leave=False):
                input_ids = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                )["input_ids"].to(device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Validation", leave=False):
                    input_ids = self.tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True
                    )["input_ids"].to(device)
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    outputs = self.model(input_ids=input_ids, labels=labels)
                    val_loss = outputs.loss
                    running_val_loss += val_loss.item()

            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))
            pbar.set_postfix(
                {"train_loss": train_losses[-1], "val_loss": val_losses[-1]}
            )

            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("training_curve.png")

        print(f"Saving to {args.output_dir}.")
        self.model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="./a1_rnn_model")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use a subset of the data for faster debugging.",
    )
    args = parser.parse_args()

    if args.task == "tokenizer":
        tokenizer_path = "tokenizer.pkl"
        if not os.path.exists(tokenizer_path):
            print("Tokenizer not found. Building a new one.")
            tokenizer = build_tokenizer(os.path.join(args.data_dir, "train.txt"))
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}.")
        else:
            print(f"Loading tokenizer from {tokenizer_path}.")
            tokenizer = A1Tokenizer.from_file(tokenizer_path)

        # Sanity check
        test_texts = ["This is a test.", "Another test.", "Unknownword!"]
        test_out = tokenizer(
            test_texts, return_tensors="pt", padding=True, truncation=True
        )
        print(test_texts)
        print(test_out)

    elif args.task == "train":
        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        config = A1RNNModelConfig(
            vocab_size=len(tokenizer),
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
        )
        model = A1RNNModel(config)
        dataset = load_dataset(
            "text",
            data_files={
                "train": os.path.join(args.data_dir, "train.txt"),
                "validation": os.path.join(args.data_dir, "val.txt"),
            },
        )
        dataset = dataset.filter(lambda x: x["text"].strip() != "")

        trainer = A1Trainer(
            model, args, dataset["train"], dataset["validation"], tokenizer
        )
        trainer.train()
