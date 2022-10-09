import json
import random
import typing

import pytorch_lightning
import transformers
import typer
from sentence_transformers import SentenceTransformer
from smart_open import smart_open
from torch.utils import data

from model import Decoder, Encoder, Retro, TextInput


class Dataset(data.Dataset):
    def __init__(self, data: typing.List[typing.Dict[str, str]], sequence_length: int,
                 tokenizer: transformers.BertTokenizer):
        self.data = data
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.random = random.Random(0)

    def __getitem__(self, item):
        text = self.data[item]["src"]
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if tokens.size(0) < self.sequence_length + 1:
            return self[self.random.randint(0, len(self) - 1)]
        start = self.random.randint(0, tokens.size(0) - self.sequence_length - 1)
        src = tokens[start: start + self.sequence_length]
        tgt = tokens[start + 1: start + self.sequence_length + 1]
        src = self.tokenizer.decode(src)
        tgt = self.tokenizer.decode(tgt)
        return TextInput(content=src, doc_id=self.data[item]["doc_id"]), tgt

    def __len__(self):
        return len(self.data)


def main(embedding_model_name: str = "bert-large-uncased", tokenizer: str = "gpt2",
         encoder_features: int = 896, encoder_depth: int = 6, encoder_heads: int = 16,
         decoder_features: int = 896, decoder_depth: int = 12, decoder_heads: int = 16,
         retrieval_frequency: int = 3, first_retrieve_at_depth: int = 6,
         query_chunk_size: int = 64, corpus_chunk_size: int = 128, topk: int = 1,
         dropout_rate: float = 0.25,
         learning_rate: float = 2e-4, weight_decay: float = 0., batch_size: int = 128, sequence_length: int = 3072,
         train_dataset_path: str = typer.Option(...), dataset_path: str = typer.Option(...)
         ):
    """
    See https://arxiv.org/pdf/2112.04426.pdf#subsection.C.1 for official configurations
    :param embedding_model_name: Name of model (on HuggingFace Hub) to use for embeddings
    :param tokenizer: Either path to local HuggingFace tokenizer or name of tokenizer from the HuggingFace Hub
    :param encoder_features: Width of the encoder's residual path
    :param encoder_depth: Number of blocks (FeedForward + Attention = 1 Block) in encoder
    :param encoder_heads: Number of attention heads used during encoder attention
    :param decoder_features: Width of the decoder's residual path
    :param decoder_depth: Number of blocks (FeedForward + ChunkedCrossAttention + Attention = 1 Block) in decoder
    :param decoder_heads: Number of attention heads used during decoder attention and decoder cross attention
    :param retrieval_frequency: Every how many blocks `ChunkedCrossAttention` should be used
    :param first_retrieve_at_depth: After which index (inclusive, starts at 1) the model should perform cross attention
    :param query_chunk_size: Number of tokens to be grouped into a "chunk" used to retrieve similar documents.
    :param corpus_chunk_size: Number of tokens to group in the corpus (-> less = more accurate, more = lower cost)
    :param topk: Number of top semantic query-document matches to retrieve
    :param dropout_rate: dropout rate to regularize the decoder (not used in encoder)
    :param learning_rate: Maximal step size used for updates during training
    :param weight_decay: Percent to of parameter to remove at every step. Relative to learning rate,
    so learning_rate=1e-4 + weight_decay=0.1 would remove 1e-5 or 0.01% of the parameter at every step.
    :param batch_size: Number of samples seen per training step
    :param sequence_length: Number of tokens per training sequence
    :param train_dataset_path: Path to json file containing training examples like {"src": <text>, "id": <text>}
    :param dataset_path: Path to jsonl file containing training examples like {"src": <text>, "id": <text>} (can be the
    same as train_dataset_path)
    """
    tokenizer: transformers.BertTokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
    embedding_model = SentenceTransformer(embedding_model_name)

    encoder = Encoder(encoder_features, encoder_heads, encoder_depth, tokenizer.vocab_size, sequence_length, 0)
    retro_at = [i for i in range(1, decoder_depth + 1)
                if (i - first_retrieve_at_depth) % retrieval_frequency == 0 and i >= first_retrieve_at_depth]
    decoder = Decoder(decoder_features, decoder_heads, decoder_depth, tokenizer.vocab_size, sequence_length, retro_at,
                      query_chunk_size, dropout_rate)

    retrieval_dataset = []  # [{"src": <text>, "id": <text>}, ...]
    with smart_open(dataset_path, mode='r') as f:
        for line in f:
            retrieval_dataset.append(json.loads(line))

    retro = Retro(embedding_model, encoder, decoder, tokenizer, retrieval_dataset, corpus_chunk_size, query_chunk_size,
                  topk, learning_rate, weight_decay)

    train_dataset = []  # [{"src": <text>, "id": <text>}, ...]
    with smart_open(train_dataset_path, mode='r') as f:
        for line in f:
            train_dataset.append(json.loads(line))
    train_dataset = Dataset(train_dataset, sequence_length, tokenizer)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size)

    trainer = pytorch_lightning.Trainer()
    trainer.fit(retro, train_dataloader)


if __name__== "__main__":
    typer.run(main)
