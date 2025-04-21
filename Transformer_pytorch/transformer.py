import copy
import torch
import math
import altair as alt
import torch.nn as nn
import pandas as pd

from torch.nn.functional import log_softmax, pad


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequence."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step.
    这里线性层映射的维度:
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core coder is a stack of N layers"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input adn mask through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward network"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow figure left for connections"""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward."""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow figure right for connections"""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scale dot Product Attention"""
    d_k = query.size(-1)
    scaling_factor = torch.sqrt(
        query.new_tensor(d_k, dtype=torch.float32)  # 使用new_tensor保持设备一致性
    )
    scores = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
    # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiheadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    cy = copy.deepcopy
    attn = MultiheadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, cy(attn), cy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, cy(attn), cy(attn), cy(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), cy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), cy(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)  # [1, 10, 512]
    ys = torch.zeros(1, 1).type_as(src)  # [1, 1]
    tgt_mask = None
    for i in range(9):
        tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        out = test_model.decode(memory, src_mask, ys, tgt_mask)  # [1, 1, 512]
        prob = test_model.generator(out[:, -1])  # [1, 11]
        _, next_word = torch.max(prob, dim=1)  # [1]
        next_word = next_word.data[0]  #
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


class EncoderWrapper(nn.Module):
    """包装编码器部分"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src, src_mask):
        return self.model.encode(src, src_mask)


class DecoderStepWrapper(nn.Module):
    """包装单步解码器部分"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.generator = model.generator

    def forward(self, memory, src_mask, ys, tgt_mask):
        # 单步解码
        out = self.model.decode(memory, src_mask, ys, tgt_mask)
        # 生成预测概率
        prob = self.generator(out[:, -1])  # 只取最后一个时间步
        return prob


def export_transformer():
    # 1. 创建并初始化模型
    test_model = make_model(11, 11, 2)
    test_model.eval()

    # 2. 准备示例输入
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    ys = torch.zeros(1, 1).type_as(src)  # 初始解码输入 [1, 1]
    tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data)  # [1, 1, 1]

    # 3. 导出编码器部分
    encoder_wrapper = EncoderWrapper(test_model)
    torch.onnx.export(
        encoder_wrapper,
        (src, src_mask),
        "transformer_encoder.onnx",
        input_names=["src", "src_mask"],
        output_names=["memory"],
        dynamic_axes={
            "src": {1: "src_seq_len"},
            "src_mask": {2: "src_seq_len"},
            "memory": {1: "src_seq_len"},
        },
        opset_version=13,
    )

    # 4. 导出单步解码器部分
    decoder_wrapper = DecoderStepWrapper(test_model)
    torch.onnx.export(
        decoder_wrapper,
        (encoder_wrapper(src, src_mask), src_mask, ys, tgt_mask),
        "transformer_decoder_step.onnx",
        input_names=["memory", "src_mask", "tgt", "tgt_mask"],
        output_names=["prob"],
        dynamic_axes={
            "memory": {1: "src_seq_len"},
            "src_mask": {2: "src_seq_len"},
            "tgt": {1: "tgt_seq_len"},
            "tgt_mask": {1: "tgt_seq_len", 2: "tgt_seq_len"},
        },
        opset_version=13,
    )
    print("ONNX模型导出成功: transformer_encoder.onnx 和 transformer_decoder_step.onnx")


def run_tests():
    for _ in range(10):
        inference_test()


if __name__ == "__main__":
    export_transformer()
    # inference_test()
