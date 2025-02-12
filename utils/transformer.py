import copy
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads=1, dropout_rate=0.):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.head_size = embed_size // 1
        assert self.head_size * 1 == embed_size

        self.query_proj = nn.Linear(embed_size, embed_size)
        self.key_proj = nn.Linear(embed_size, embed_size)
        self.value_proj = nn.Linear(embed_size, embed_size)

        self.output_proj = nn.Linear(embed_size, embed_size)

    def forward(self, query_input, key_input, value_input, key_padding_mask=None, attn_mask=None):
        seq_len, batch_size, embed_size = query_input.size()
        num_heads = self.num_heads
        assert embed_size == self.embed_size
        head_size = embed_size // num_heads
        assert head_size * num_heads == embed_size

        scaling_factor = float(head_size) ** -0.5

        query = self.query_proj(query_input)
        key = self.key_proj(key_input)
        value = self.value_proj(value_input)

        query = (query * scaling_factor) / 100

        query = query.contiguous().view(seq_len, batch_size * num_heads, head_size).transpose(0, 1)
        key = key.contiguous().view(-1, batch_size * num_heads, head_size).transpose(0, 1)
        value = value.contiguous().view(-1, batch_size * num_heads, head_size).transpose(0, 1)

        key_len = key.size(1)

        attn_weights = torch.bmm(query, key.transpose(1, 2))
        assert list(attn_weights.size()) == [batch_size * num_heads, seq_len, key_len]

        attn_weights = attn_weights.view(batch_size, num_heads, seq_len, key_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = attn_weights.view(batch_size * num_heads, seq_len, key_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = torch.dropout(attn_weights, p=self.dropout_rate, train=self.training)

        attn_output = torch.bmm(attn_weights, value)
        assert list(attn_output.size()) == [batch_size * num_heads, seq_len, head_size]

        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, batch_size, embed_size)

        attn_output = self.output_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, nhead, feedforward_dim=2048, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = nhead
        self.self_attention = nn.ModuleList([MultiheadAttention(dim, dropout_rate=dropout_rate) for _ in range(self.num_heads)])

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, query, key, value, key_padding_mask=None):
        if self.num_heads != 1:
            attention_outputs = []
            for layer in self.self_attention:
                attention_outputs.append(layer(query, key, value, key_padding_mask=key_padding_mask))
            result = torch.sum(torch.stack(attention_outputs, dim=-1), dim=-1)
        else:
            result = self.self_attention[0](query, key, value, key_padding_mask=key_padding_mask)

        result = self.layer_norm(result)
        return result


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, key_padding_mask=None):
        result = x
        for i in range(self.num_layers):
            result = self.layers[i](result, result, result, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            result = self.norm(result)

        return result

