import dataclasses
import typing

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F


@dataclasses.dataclass
class TokenInput:
    content: torch.Tensor
    doc_id: str


class Database(nn.Module):
    def __init__(self, model: SentenceTransformer, texts: typing.List[typing.Dict[str, str]],
                 corpus_chunk_size: int = 128,
                 query_chunk_size: int = 64, topk: int = 1):
        super(Database, self).__init__()
        self.model = model
        self.corpus_chunk_size = corpus_chunk_size
        self.query_chunk_size = query_chunk_size
        self.topk = topk

        self.tokenizer: transformers.BertTokenizer = model.tokenizer
        self.special_tokens = len(self.tokenizer("A")["input_ids"]) - 1

        out = zip(*[self._embed(txt["src"], corpus_chunk_size - query_chunk_size) + (txt["id"],) for txt in texts])
        embeddings, chunks, ids = out
        self.chunks = [t for doc in chunks for t in doc]
        self.ids = {doc_id: list_index for list_index, doc_id in enumerate(ids)}
        self.lengths = np.array([embd.size(0) for embd in embeddings])
        self.cumulative_lengths = np.cumsum(self.lengths, 0)
        self.register_buffer("embeddings", torch.cat(embeddings, 0).transpose(1, 0))

    def _batch(self, tokens: torch.Tensor, keep: int):
        offset = tokens.size(0) % self.query_chunk_size
        chunks = tokens[:tokens.size(0) - offset].view(-1, self.query_chunk_size)[:, :keep]
        if offset > keep:
            residual = tokens[tokens.size(0) - offset:][:keep]
            if keep > residual.size(0):
                if self.tokenizer.pad_token_id is not None:
                    token = self.tokenizer.pad_token_id
                elif self.tokenizer.eos_token_id is not None:
                    token = self.tokenizer.eos_token_id
                else:
                    token = 0
                pad = torch.full([keep - residual.size(0)], token, dtype=residual.dtype, device=residual.device)
                residual = torch.cat([residual, pad], 0)
            chunks = torch.cat([chunks, residual.reshape(1, -1)], 0)
        return chunks

    def _embed(self, text: str, skip: int):
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        chunks = self._batch(tokens, skip)
        return self.embed(chunks), chunks

    def embed(self, tokens: torch.Tensor):
        inp = {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
        return F.normalize(self.model(inp)["sentence_embedding"], 1)

    def forward(self, inp: TokenInput) -> typing.List[typing.List[str]]:
        query, doc_id = inp.content, self.ids.get(inp.doc_id)
        start = self.cumulative_lengths[doc_id]
        if doc_id is None:
            embeddings = self.embeddings  # if doc_id is outside the known range, it's not an issue
        elif doc_id == 0:
            embeddings = self.embeddings[:, start:]
        elif doc_id == self.lengths.shape[0] - 1:
            embeddings = self.embeddings[:, :start]
        elif 0 < doc_id < self.lengths.shape[0] - 1:  # can't concat zero-sized tensor, so have to guard case with ifs
            end = self.cumulative_lengths[doc_id + 1]
            embeddings = torch.cat([self.embeddings[:, :start], self.embeddings[:, end:]], 1)
        else:
            raise ValueError(f"Unknown {doc_id=}")

        similarity = self.embed(self._batch(query, self.query_chunk_size)) @ embeddings
        # paper uses l2-distance, but expensive in torch

        values, indices = torch.topk(similarity, self.topk, 1)
        if doc_id is None:
            pass
        elif doc_id == 0:
            indices += start
        elif doc_id == self.lengths.shape[0] - 1:
            pass
        elif 0 < doc_id < self.lengths.shape[0] - 1:  # can't concat zero-sized tensor, so have to guard case with ifs
            indices += torch.where(indices >= start, end - start, 0)
        else:
            raise ValueError(f"Unknown {doc_id=}")
        return [[self.chunks[i] for i in qry] for qry in indices.cpu().tolist()]


class RMSNorm(nn.Module):
    """
    From `Root Mean Square Layer Normalization` (https://arxiv.org/abs/1910.07467)
    Significantly better than LayerNorm at scale (see https://github.com/HomebrewNLP/HomebrewNLP-Jax/pull/77)
    """

    def __init__(self, features: int):
        super(RMSNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, inp: torch.Tensor):
        return F.normalize(inp, 2) * self.scale


class SandwichNorm(nn.Module):
    """
    RETRO doesn't mention whether it uses pre-, post- or sandwich-normalization.
    Using SandwichNorm from https://arxiv.org/abs/2102.11382 as it's more stable at scale
    """

    def __init__(self, features: int, module: torch.nn.Module):
        super(SandwichNorm, self).__init__()
        self.in_norm = RMSNorm(features)
        self.core = module
        self.out_norm = RMSNorm(features)

    def forward(self, inp: torch.Tensor, *other: torch.Tensor):
        return self.out_norm(self.core(self.in_norm(inp), *other))


class CrossAttention(nn.Module):
    def __init__(self, features: int, masked: bool, heads: int):
        self.masked = masked
        self.heads = heads
        self.features_per_head = features // heads
        super(CrossAttention, self).__init__()
        self.to_q = nn.Linear(features, features)
        self.to_kv = nn.Linear(features, features * 2)
        self.to_out = nn.Linear(features, features)

    def forward(self, attend_from: torch.Tensor, attend_to: torch.Tensor):
        batch, sequence, features = attend_from.size()
        qry = self.to_q(attend_from).view(batch, sequence, self.heads, self.features_per_head)
        key, val = self.to_kv(attend_to).chunk(2, -1)
        key = key.view(batch, -1, self.heads, self.features_per_head)
        val = val.view(batch, -1, self.heads, self.features_per_head)

        logits = torch.einsum("bshf,bzhf->bhsz", qry, key)
        if self.masked:
            if attend_to.size(1) != attend_from.size(1):
                raise ValueError("Can't do autoregressive attention if sizes are not equal.")
            logits = logits - torch.triu(torch.ones(attend_to.size(1), attend_to.size(1))) * 1e12
        attention_map = torch.softmax(logits, 3)

        out = torch.einsum("bhsz,bzhf->bshf", attention_map, val).reshape(batch, sequence, features)
        return self.to_out(out)  # Shape[Batch, Sequence, Features]


class SelfAttention(CrossAttention):
    def forward(self, attend_from: torch.Tensor):
        return super(SelfAttention, self).forward(attend_from, attend_from)


class ChunkedCrossAttention(CrossAttention):
    def __init__(self, features: int, masked: bool, heads: int, query_chunk_size: int):
        self.query_chunk_size = query_chunk_size
        super(ChunkedCrossAttention, self).__init__(features, masked, heads)

    def forward(self, attend_from: torch.Tensor, attend_to: torch.Tensor):
        batch, sequence, features = attend_from.size()
        src = attend_from[:, self.query_chunk_size - 1:-1].reshape(-1, self.query_chunk_size, features)
        out = super(ChunkedCrossAttention, self).forward(src, attend_to)
        out = out.view(batch, -1, features)
        return torch.cat([attend_from[:, :self.query_chunk_size - 1], out, attend_from[:, -1:]], 1)


class FeedForward(nn.Module):  # Not clarified in paper
    def __init__(self, features: int, expansion: int, dropout_rate: float):
        super(FeedForward, self).__init__()
        self.to_intermediate = nn.Linear(features, features * expansion)
        self.activation = nn.GELU()
        self.normalization = RMSNorm(features * expansion)
        self.dropout = nn.Dropout(dropout_rate)
        self.from_intermediate = nn.Linear(features * expansion, features)

    def forward(self, x: torch.Tensor):
        x = self.to_intermediate(x)
        x = self.activation(x)
        x = self.normalization(x)
        x = self.dropout(x)
        return self.from_intermediate(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, heads: int, retrieves: bool, query_chunk_size: int, dropout_rate: float):
        super(DecoderBlock, self).__init__()
        self.attn = SandwichNorm(features, SelfAttention(features, masked=True, heads=heads))
        if retrieves:
            cca = ChunkedCrossAttention(features, heads=heads, masked=False, query_chunk_size=query_chunk_size)
            self.cca = SandwichNorm(features, cca)
        else:
            self.cca = None
        self.ffw = SandwichNorm(features, FeedForward(features, 4, dropout_rate))

    def forward(self, inputs: typing.Tuple[torch.Tensor, torch.Tensor]):
        inp, neighbors = inputs
        inp = inp + self.attn(inp)
        if self.cca is not None:
            inp = inp + self.cca(inp, neighbors)
        return inp + self.ffw(inp), neighbors


class EncoderBlock(nn.Module):
    """
    Not specified in the paper. Assuming a typical transformer encoder.
    """

    def __init__(self, features: int, heads: int, dropout_rate: float):
        super(EncoderBlock, self).__init__()
        self.attn = SandwichNorm(features, SelfAttention(features, masked=False, heads=heads))
        self.ffw = SandwichNorm(features, FeedForward(features, 4, dropout_rate))

    def forward(self, inp: torch.Tensor):
        inp = inp + self.attn(inp)
        return inp + self.ffw(inp)


class Embedding(nn.Module):
    """
    Combined position- and token-embedding
    RETRO uses relative position embeddings, but didn't specify which kind. As absolute position embeddings
    outperform relative position embeddings, they're used here.
    """

    def __init__(self, vocab: int, features: int, sequence_length: int):
        super(Embedding, self).__init__()
        self.input_embedding = nn.Embedding(vocab, features)
        self.position_embedding = nn.Parameter(torch.randn(sequence_length, features) / features ** 0.5)

    def forward(self, input_ids: torch.Tensor):
        return self.input_embedding(input_ids) + self.position_embedding[None, :input_ids.size(1)]


class Encoder(nn.Module):
    """
    Not specified in the paper. Assuming a typical transformer encoder.
    """

    def __init__(self, features: int, heads: int, depth: int, vocab: int, sequence_length: int, dropout_rate: float):
        super(Encoder, self).__init__()
        self.embedding = Embedding(vocab, features, sequence_length)
        self.core = nn.Sequential(*[EncoderBlock(features, heads, dropout_rate) for _ in range(depth)])

    def forward(self, input_ids: torch.Tensor):
        return self.core(self.embedding(input_ids))


class Decoder(nn.Module):
    def __init__(self, features: int, heads: int, depth: int, vocab: int, sequence_length: int,
                 retro_at: typing.List[int], query_chunk_size: int, dropout_rate: float):
        super(Decoder, self).__init__()
        self.embedding = Embedding(vocab, features, sequence_length)
        self.core = nn.Sequential(*[DecoderBlock(features, heads, i in retro_at, query_chunk_size, dropout_rate)
                                    for i in range(depth)])
        self.output = nn.Linear(features, vocab)

    def forward(self, input_ids: torch.Tensor, neighbor_embeddings: torch.Tensor):
        inp = self.embedding(input_ids)
        inp, _unmodified_neighbor_embeddings = self.core((inp, neighbor_embeddings))
        return self.output(inp)


class Retro(pl.LightningModule):
    def __init__(self, embedding_model: SentenceTransformer, encoder: Encoder, decoder: Decoder,
                 tokenizer: transformers.BertTokenizer, texts: typing.List[typing.Dict[str, str]],
                 corpus_chunk_size: int = 128, query_chunk_size: int = 64, topk: int = 1, learning_rate: float = 1e-4,
                 weight_decay=0.1):
        super(Retro, self).__init__()
        self.corpus_chunk_size = corpus_chunk_size

        self.database = Database(embedding_model, texts, corpus_chunk_size, query_chunk_size, topk)
        self.encoder = encoder
        self.decoder = decoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, inp: typing.Dict[str, typing.Union[typing.List[torch.Tensor], typing.List[int]]]):
        inp = TokenInput(**inp)
        retrieved = [[self.database(TokenInput(content=i, doc_id=d))] for i, d in zip(inp.content, inp.doc_id)]
        retrieved = torch.stack([torch.stack([torch.stack([torch.stack(a) for a in b]) for b in c]) for c in retrieved])
        retrieved = retrieved.reshape(len(inp.doc_id), -1, self.database.topk, retrieved.size(-1))
        retrieved = retrieved[:, :-1]  # [Batch, Chunks, TopK, Tokens] -> [Batch, Chunks - 1, TopK, Tokens]
        # [Batch, Chunks - 1, TopK, Tokens]  -> [Batch * (Chunks - 1), TopK * Tokens]
        neighbors = retrieved.reshape(-1, self.database.topk * retrieved.size(-1))
        neighbor_embeddings = self.encoder(neighbors)
        return self.decoder(inp.content, neighbor_embeddings)

    def training_step(self, inp: typing.Tuple[typing.Dict[str, typing.Union[typing.List[str], typing.List[int]]],
                                              typing.List[str]]) -> torch.Tensor:
        src, target_ids = inp
        logits = self.forward(src)
        return F.cross_entropy(logits.transpose(1, 2), target_ids)

    def generate(self, src: typing.List[TokenInput], tokens: int):
        for _ in range(tokens):
            logits = self.forward(src)
            distribution = torch.distributions.Categorical(logits)
            tokens = distribution.sample([len(src)])
            for s, t in zip(src, tokens):
                s.content += self.tokenizer.decode(self.tokenizer.encode(s.content) + [t])

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        opt = torch.optim.AdamW([
                {
                        "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.weight_decay,
                        },
                {
                        "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0,
                        },
                ],
                weight_decay=0, lr=self.learning_rate, betas=(0.9, 0.95))
        return opt
