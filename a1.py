import matplotlib.pyplot as plt
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext
import numpy as np
import argparse
import tqdm
import sys, time, os

from a2 import A2Transformer, A2ModelConfig


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
    num_lines = sum(1 for _ in open(train_file, "r"))
    with open(train_file, "r") as file:
        for line in tqdm.tqdm(file, desc="Building tokenizer", disable=args.no_tqdm, total=num_lines):
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
    for i, (word, _) in enumerate(counter.most_common(max_voc_size - 4), start=4):
        str_to_int[word] = i
        int_to_str[i] = word

    total = counter.total()
    sorted_freqs = [count / total for _, count in counter.most_common()]
    coverage = np.sum([freq for freq in sorted_freqs[:len(str_to_int)]])
    print(f"Total words in training set: {total}")
    print(f"Vocabulary size: {len(str_to_int)}")
    print(f"Vocabulary coverage: {coverage:.2%}")

    plt.plot(sorted_freqs)
    plt.xlabel("Token rank")
    plt.ylabel("Token frequency")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.savefig("token_freqs.png", dpi=200)
    plt.close()

    plt.plot(np.cumsum(sorted_freqs))
    plt.xlabel("Token rank")
    plt.ylabel("Cumulative frequency")
    plt.xscale("log")
    plt.grid(alpha=0.3)
    plt.savefig("token_cumulative_freqs.png", dpi=200)
    plt.close()

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

    def __init__(self, vocab_size=0, embedding_size=0, hidden_size=0, num_layers=0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers


class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""

    config_class = A1RNNModelConfig
    _dynamic_tied_weights_keys = ["unembedding.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.GRU(
            config.embedding_size,
            config.hidden_size,
            config.num_layers,
            dropout=args.dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(args.dropout)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Tie weights and initialize
        self.tie_weights()
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
    
    def tie_weights(self):
        self.unembedding.weight = self.embedding.weight
    
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
        logits = self.unembedding(self.dropout(rnn_out))
        if labels is not None:
            # Logits are (batch_size, seq_length, vocab_size) and labels are (batch_size, seq_length)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = None

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
        self.model.to(device)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.

        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if name in ["embedding.weight", "unembedding.weight"]:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [{"params": no_decay, "weight_decay": 0.0}, 
            {"params": decay, "weight_decay": args.weight_decay}],
            lr=args.learning_rate,
            betas=(0.9, 0.9),
            fused=True if "cuda" in device.type else False
        )

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

        # Schedule cosine decay
        total_steps = len(train_loader) * args.num_train_epochs
        
        def schedule_fn(current_step):
            """Linear warmup for the first 10% of steps, then cosine decay."""
            if current_step < 0.1 * total_steps:
                return current_step / (0.1 * total_steps)
            else:
                return 0.5 * (1 + np.cos(np.pi * (current_step - 0.1 * total_steps) / (0.9 * total_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_fn)

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

        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        val_perplexities = []
        pbar = tqdm.trange(args.num_train_epochs, desc="Epoch", disable=args.no_tqdm)
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for batch in tqdm.tqdm(train_loader, desc="Batch", leave=False, disable=args.no_tqdm):
                tokenized = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                input_ids = tokenized["input_ids"].to(device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if not args.full_precision else nullcontext():
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()

            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Validation", leave=False, disable=args.no_tqdm):
                    tokenized = self.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                    input_ids = tokenized["input_ids"].to(device)
                    labels = input_ids.clone()
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if not args.full_precision else nullcontext():
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        val_loss = outputs.loss

                    running_val_loss += val_loss.item()

            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))
            val_perplexities.append(np.exp(val_losses[-1]))
            pbar.set_postfix({
                "loss": train_losses[-1], 
                "val_loss": val_losses[-1], 
                "val_perp": val_perplexities[-1],
                "lr": scheduler.get_last_lr()[0]
            })
            if args.no_tqdm:
                print(f"Epoch {epoch + 1}/{args.num_train_epochs}: train_loss={train_losses[-1]:.4f}, val_loss={val_losses[-1]:.4f}, val_perp={val_perplexities[-1]:.2f}, lr={scheduler.get_last_lr()[0]:.2e}")

            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig("training_curve.png", dpi=200)
            plt.close()

            plt.plot(val_perplexities)
            plt.xlabel("Epoch")
            plt.ylabel("Perplexity")
            plt.grid(alpha=0.3)
            plt.savefig("perplexity_curve.png", dpi=200)
            plt.close()

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                self.model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, choices=["tokenizer", "train", "predict", "generate", "nn", "pca", "tsne"], help="The task to perform.")
    parser.add_argument("prompt", type=str, nargs="?", default="", help="The prompt to use for prediction or generation tasks.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="trainer_output")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--max_voc_size", type=int, default=25_000)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true", help="Use a subset of the data for faster debugging.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA even if it is available.")
    parser.add_argument("--no_tqdm", action="store_true", help="Whether to disable tqdm progress bars.")
    parser.add_argument("--use_cpu", action="store_true", help="Force the trainer to use the CPU.")
    parser.add_argument("--full_precision", action="store_true", help="Whether to disable mixed precision training (if applicable).")
    parser.add_argument("--a2", action="store_true", help="Whether to use the A2 model instead of the A1 model.")
    args = parser.parse_args()

    if args.a2:
        Model = A2Transformer
        ModelConfig = A2ModelConfig
    else:
        Model = A1RNNModel
        ModelConfig = A1RNNModelConfig

    if args.task == "tokenizer":
        tokenizer_path = "tokenizer.pkl"
        if not os.path.exists(tokenizer_path):
            print("Tokenizer not found. Building a new one.")
            tokenizer = build_tokenizer(
                os.path.join(args.data_dir, "train.txt"), 
                max_voc_size=args.max_voc_size, 
                model_max_length=args.model_max_length
            )
            tokenizer.save(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}.")
        else:
            print(f"Loading tokenizer from {tokenizer_path}.")
            tokenizer = A1Tokenizer.from_file(tokenizer_path)

        # Sanity check
        test_texts = ["This is a test.", "Another test.", "pear!"]
        test_out = tokenizer(
            test_texts, return_tensors="pt", padding=True, truncation=True
        )
        print(test_texts)
        print(test_out)

    elif args.task == "train":
        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        config = ModelConfig(
            vocab_size=len(tokenizer),
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
        )
        model = Model(config)
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

    elif args.task == "predict":
        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        model = Model.from_pretrained(args.output_dir)
        model.eval()

        # Predict the next token for a test sentence
        test_sentence = [args.prompt]
        input_ids = tokenizer(test_sentence, return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :-1] # Drop <EOS> token for prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # Print the top k predicted tokens
            k = 5
            top_k_indices = torch.topk(next_token_logits, k).indices
            top_k_tokens = [tokenizer.int_to_str[idx.item()] for idx in top_k_indices]
            print(f"Input: {test_sentence[0]}")
            print(f"Top {k} predicted next tokens: {top_k_tokens}")

    elif args.task == "generate":
        
        def top_p_sample(logits, p=0.9, temperature=1.0):
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            logits[sorted_indices[sorted_indices_to_remove]] = -float("inf")
            sample = torch.multinomial(torch.softmax(logits, dim=-1) / temperature, num_samples=1)
            return sample.unsqueeze(0)
        
        def top_k_sample(logits, k=10, temperature=1.0):
            topk_logits, topk_indices = torch.topk(logits, k)
            idx = torch.multinomial(torch.softmax(topk_logits, dim=-1) / temperature, num_samples=1)
            sample = topk_indices[idx]
            return sample.unsqueeze(0)        
        
        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        model = Model.from_pretrained(args.output_dir)
        model.eval()

        # Generate text autoregressively
        prompt = args.prompt
        input_ids = tokenizer([prompt], return_tensors="pt")["input_ids"]
        input_ids = input_ids[:, :-1] # Drop <EOS> token for generation
        max_gen_length = args.model_max_length - input_ids.shape[1]
        generated_ids = input_ids
        generated_text = []
        print(f"Prompt: {prompt}")
        print(f"Generated continuation:", end=" ", flush=True) 
        with torch.no_grad():
            for _ in range(max_gen_length):
                outputs = model(input_ids=generated_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]

                # Don't generate <UNK>, <PAD>, or <BOS>
                next_token_logits[tokenizer.unk_token_id] = -float("inf")
                next_token_logits[tokenizer.pad_token_id] = -float("inf")
                next_token_logits[tokenizer.bos_token_id] = -float("inf")

                # Sample randomly
                #next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                #next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).unsqueeze(0)
                next_token_id = top_k_sample(next_token_logits)               
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                if next_token_id.item() == tokenizer.eos_token_id:
                    print()
                    break
                print(tokenizer.int_to_str[next_token_id.item()], end=" ", flush=True)
        

    elif args.task == "nn":
        # Nearest neighbors in the embedding space
        def nearest_neighbors(emb, voc, inv_voc, word, n_neighbors=5):
            # Look up the embedding for the test word.
            test_emb = emb.weight[voc[word]]
            
            # We'll use a cosine similarity function to find the most similar words.
            sim_func = nn.CosineSimilarity(dim=1)
            cosine_scores = sim_func(test_emb, emb.weight)
            
            # Find the positions of the highest cosine values.
            near_nbr = cosine_scores.topk(n_neighbors+1)
            topk_cos = near_nbr.values[1:]
            topk_indices = near_nbr.indices[1:]
            # NB: the first word in the top-k list is the query word itself!
            # That's why we skip the first position in the code above.
            
            # Finally, map word indices back to strings, and put the result in a list.
            result = [(inv_voc[ix.item()], cos.item()) for ix, cos in zip(topk_indices, topk_cos)]
            result.reverse() # Sort from most similar to least similar
            return result

        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        model = Model.from_pretrained(args.output_dir)
        model.eval()

        word = args.prompt.strip().lower()
        print(f"Nearest neighbors for '{word}':")
        nn = nearest_neighbors(model.embedding, tokenizer.str_to_int, tokenizer.int_to_str, word)
        for nbr, score in nn:
            print(f"{nbr}: {score:.2e}")

    elif args.task == "pca" or args.task == "tsne":
        def plot_embeddings(emb, voc, words, method="pca"):
            vectors = np.vstack([emb.weight[voc[w]].cpu().detach().numpy() for w in words])
            vectors -= vectors.mean(axis=0)
            if method == "pca":
                twodim = TruncatedSVD(n_components=2).fit_transform(vectors)
            elif method == "tsne":
                twodim = TSNE(n_components=2, init="pca", perplexity=3).fit_transform(vectors)
            plt.figure(figsize=(5,5))
            plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
            for word, (x,y) in zip(words, twodim):
                plt.text(x+0.02, y, word)
            plt.axis('off')
            plt.savefig(f"{method}_scatter.png", dpi=200)
            plt.close()

        tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
        model = Model.from_pretrained(args.output_dir)
        model.eval()

        words = ['sweden', 'denmark', 'europe', 'africa', 'london', 'stockholm', 'large', 'small', 'great', 'black', '3', '7', '10', 'seven', 'three', 'ten', '1984', '2005', '2010']
        plot_embeddings(model.embedding, tokenizer.str_to_int, words, method=args.task)