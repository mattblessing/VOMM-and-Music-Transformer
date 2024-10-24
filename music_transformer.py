"""
This code is adapted from the GitHub repository at: https://github.com/spectraldoy/music-transformer.
"""
from vocabulary import pad_token_index
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


"""
Masking functionality for the Music Transformer.
"""


def create_padding_mask(batch, n=4):
    """
    Create a mask with ones where a padding token appears in each sequence in `batch`.
    Params:
    - torch.Tensor batch: an unembedded batch of input sequences of shape (batch_size, seq_len)
    - int n: number of dimensions to broadcast the mask to
    Return:
    - torch.Tensor broadcasted_mask: a tensor of ones where the padding token appears in the batch of shape (batch_size, 1, ..., 1, seq_len)
    """
    # Find positions equal to pad_token
    mask = torch.eq(batch, pad_token_index).float()

    # Add extra dimensions
    broadcasted_mask = mask.view(*mask.shape[:-1], *[1 for _ in range(n-2)], mask.shape[-1]).to(device)

    return broadcasted_mask


def create_look_ahead_mask(seq_len):
    """
    Create an upper triangular mask of ones for the calculation of scaled dot-product attention to prevent the transformer
    looking ahead at future tokens, so the next model output is based only on the current and previous tokens in the input sequence.
    Params:
    - int seq_len: the input sequence length
    Return:
    - torch.Tensor mask: an upper triangular mask of ones of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).float().to(device)
    return mask


def create_mask(batch, n=4):
    """
    Create a combined padding and look ahead mask for the input batch.
    Params:
    - torch.Tensor batch: an unembedded batch of input sequences of shape (batch_size, seq_len)
    - int n: number of dimensions to broadcast the mask to
    Return:
    - torch.Tensor combined_mask: the combined padding and look ahead mask of shape (batch_size, 1, ..., 1, seq_len, seq_len)
    """
    padding_mask = create_padding_mask(batch, n)
    look_ahead_mask = create_look_ahead_mask(batch.shape[-1])
    combined_mask = torch.max(padding_mask, look_ahead_mask)

    return combined_mask


"""
Layers for the Music Transformer.
"""


def abs_positional_encoding(max_position, hidden_dim, n=3):
    """
    Get sinusoidal absolute positional encodings of shape `hidden_dim` for `max_position` positions.
    Params:
    - int max_position: maximum position for which to calculate positional encoding
    - int hidden_dim: the hidden dimension size of the transformer
    - int n: number of dimensions to broadcast the positional encodings to
    Return:
    - torch.Tensor embeddings: the positional encodings of shape (1, ..., 1, max_position, hidden_dim)
    """
    # Get set of all positions to consider
    positions = torch.arange(max_position).float().to(device)

    # Get angles to input to sinusoid functions
    k = torch.arange(hidden_dim).float().to(device)
    coeffs = 1 / torch.pow(10000, 2 * (k // 2) / hidden_dim)
    angles = positions.view(-1, 1) @ coeffs.view(1, -1)

    # Apply sin to the even indices of the angles along the last axis of the encodings
    angles[:, 0::2] = torch.sin(angles[:, 0::2])
    # Apply cos to the odd indices of the angles along the last axis of the encodings
    angles[:, 1::2] = torch.cos(angles[:, 1::2])

    # Add extra dimensions
    encodings = angles.view(*[1 for _ in range(n-2)], max_position, hidden_dim)

    return encodings


def skew(t):
    """
    Skewing procedure to move the relative logits to their correct positions.
    Params:
    - torch.Tensor t: tensor to skew
    Return:
    - torch.Tensor skewed_t: the skewed version of t
    """
    # Pad the tensor with a dummy column before the leftmost column
    padded = F.pad(t, [1, 0])

    # Reshape the tensor to diagonalise the columns in the last 2 dimensions
    skewed_t = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])

    # Remove the first column
    skewed_t = skewed_t[:, 1:]

    # Reshape back to shape of t
    skewed_t = skewed_t.reshape(*t.shape)

    return skewed_t


def relative_positional_attention(q, k, v, e=None, mask=None):
    """
    Compute the relative scaled dot-product attention, which allows the transformer to attend to all relevant elements of
    the input sequences as well as the relative distances between them.
    Params:
    - torch.Tensor q: queries of shape (batch_size, num_heads, seq_len, hidden_dim / num_heads)
    - torch.Tensor k: keys of shape (batch_size, num_heads, seq_len, hidden_dim / num_heads)
    - torch.Tensor v: values of shape (batch_size, num_heads, seq_len, hidden_dim / num_heads)
    - torch.Tensor e (optional): relative position embeddings of shape (num_heads, seq_len, hidden_dim / num_heads)
    - torch.Tensor mask (optional): mask for input batch of shape (batch_size, 1, seq_len, seq_len)
    Return:
    - torch.Tensor attention: the relative positional attention probabilities of shape (batch_size, num_heads, seq_len, depth)
    """
    # Compute Q*K^t
    QKt = torch.matmul(q, k.transpose(-1, -2))  # shape (batch_size, num_heads, seq_len, seq_len)

    # Compute S^rel
    # If not using relative position embeddings
    if e is None:
        Srel = torch.zeros(*q.shape[:-1], k.shape[-2], device=device)  # shape (batch_size, num_heads, seq_len, seq_len)
    # If using relative position embeddings
    else:
        Srel = skew(torch.matmul(q, e.transpose(-1, -2)))  # shape (batch_size, num_heads, seq_len, seq_len)

    # Find and scale the attention logits
    Dh = math.sqrt(k.shape[-1])
    scaled_attention_logits = (QKt + Srel) / Dh  # shape (batch_size, num_heads, seq_len, seq_len)

    # Add scaled mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Calculate attention weights by softmaxing on the last dimension and then multiplying by v
    attention = torch.matmul(F.softmax(scaled_attention_logits, dim=-1), v)  # shape (batch_size, num_heads, seq_len, depth)

    return attention


class MultiHeadAttention(nn.Module):
    """
    A multi-head relative attention block that computes relative attention for an input batch along `num_heads` attention heads.
    """

    def __init__(self, hidden_dim, num_heads, max_rel_dist, bias=True):
        """
        Params:
        - int hidden_dim: the hidden dimension size of the transformer
        - int num_heads: the number of attention heads
        - int max_rel_dist: maximum relative distance between positions to consider in creating relative position embeddings
        (0 gives normal attention)
        - bool bias (optional): if set to True, all linear layers in the block will learn an additive bias
        """
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist
        self.batch_first = False  #  For nn.TransformerDecoder

        if hidden_dim % num_heads != 0:
            raise ValueError("`dim` must be divisible into `num_heads` heads")

        self.depth = self.hidden_dim // self.num_heads

        # Parameter matrix to generate Q from input
        self.Wq = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        # Parameter matrix to generate K from input
        self.Wk = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)
        # Parameter matrix to generate V from input
        self.Wv = nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias)

        # Relative position embeddings ordered from E_{-max_rel_dist + 1} to E_0
        self.E = nn.Embedding(self.max_rel_dist, self.hidden_dim)

        # Final output layer
        self.Wo = nn.Linear(self.hidden_dim, self.hidden_dim,
                            bias=True)

    def relative_position_embeddings(self, seq_len):
        """
        Get relative position embeddings required to calculate attention on input of length `seq_len`.
        Params:
        - int seq_len: length of input sequence
        Return:
        - torch.Tensor required_embeddings: max(0, seq_len - max_rel_dist) copies of E_{-max_rel_dist} followed by 
        E[max(0, max_rel_dist - seq_len) : max_rel_dist]
        """
        # Required relative position embeddings
        first_embedding = self.E(torch.arange(1, device=device)).clone()
        required_embeddings = torch.cat(
             # max(0, seq_len - max_rel_dist) copies of the first embedding
            [*[first_embedding.clone() for _ in range(max(0, seq_len - self.max_rel_dist))],  
             # E[max(0, max_rel_dist - seq_len) : max_rel_dist]
             self.E(torch.arange(max(0, self.max_rel_dist - seq_len), self.max_rel_dist, device=device))],  
            dim=0
        )

        return required_embeddings

    def split_heads(self, x):
        """
        Split input x along `num_heads` heads.
        Params:
        - torch.Tensor x: the input tensor to split into heads, has shape: (..., seq_len, hidden_dim)
        Return:
        - torch.Tensor x: the input tensor with shape (..., num_heads, seq_len, depth)
        """
        # Reshape and transpose x
        x = x.view(*x.shape[:-1], self.num_heads, self.depth).transpose(-2, -3)  # shape (batch_size, num_heads, seq_len, depth)

        return x

    def forward(self, q, k, v, mask=None):
        """
        Computes multi-head relative attention on input tensors Q, K, V.
        Params:
        - torch.Tensor q: queries of shape (batch_size, seq_len, hidden_dim)
        - torch.Tensor k: keys of shape (batch_size, seq_len, hidden_dim)
        - torch.Tensor v: values of shape (batch_size, seq_len, hidden_dim)
        - torch.Tensor mask (optional): mask for input batch of shape (batch_size, 1, seq_len, seq_len)
        Return:
        - torch.Tensor attention: the relative positional attention probabilities of shape (batch_size, seq_len, hidden_dim)
        """
        # Get Q, K, V
        q = self.Wq(q)  # shape (batch_size, seq_len, hidden_dim)
        k = self.Wk(k)  # shape (batch_size, seq_len, hidden_dim)
        v = self.Wv(v)  # shape (batch_size, seq_len, hidden_dim)

        # Get required relative position embeddings from E
        e = self.relative_position_embeddings(k.shape[-2])  # shape (seq_len, hidden_dim)

        # Split into heads
        q = self.split_heads(q)  # shape (batch_size, num_heads, seq_len, depth)

        k = self.split_heads(k)  # shape (batch_size, num_heads, seq_len, depth)

        v = self.split_heads(v)  # shape (batch_size, num_heads, seq_len, depth)

        e = self.split_heads(e)  # shape (num_heads, seq_len, depth)

        # Compute multi-head relative attention of shape (batch_size, num_heads, seq_len, depth)
        rel_scaled_attention = relative_positional_attention(q, k, v, e, mask=mask)

        # Concatenate attention heads and pass through final layer
        rel_scaled_attention = rel_scaled_attention.transpose(-2, -3)  # shape (batch_size, seq_len, num_heads, depth)
        attention = self.Wo(rel_scaled_attention.reshape(*rel_scaled_attention.shape[:-2], self.hidden_dim))  # shape (batch_size, seq_len, hidden_dim)

        return attention


class FeedForwardNetwork(nn.Module):
    """
    Feedforward sub-layer that follows each multi-head attention block in each transformer layer.
    """

    def __init__(self, hidden_dim, ff_hidden_dim, bias=True):
        """
        Params:
        - int hidden_dim: the hidden dimension size of the transformer
        - int ff_hidden_dim: the intermediate dimension size of the feedforward network
        - bool bias (optional): if set to True, all linear layers in the feedforward network will learn an additive bias
        """
        super(FeedForwardNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.ff_hidden_dim = ff_hidden_dim

        self.main = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, hidden_dim, bias=bias)
        )

    def forward(self, x):
        return self.main(x)


class DecoderLayer(nn.Module):
    """
    A transformer decoder layer consisting of two sublayers: a masked multi-head attention block followed by a feedforward network.
    This class is designed to be used by torch's nn.TransformerDecoder.
    """

    def __init__(self, hidden_dim, num_heads, max_rel_dist, ff_hidden_dim, bias=True, dropout=0.1, layernorm_eps=1e-6):
        """
        Params:
        - int hidden_dim: the hidden dimension size of the transformer
        - int num_heads: the number of attention heads
        - int max_rel_dist: maximum relative distance between positions to consider in creating relative position embeddings 
        (0 gives normal attention)
        - int ff_hidden_dim: the intermediate dimension size of the feedforward network
        - bool bias (optional): if set to True, all linear layers in the feedforward network will learn an additive bias
        - float dropout (optional): dropout rate for training the model
        - float layernorm_eps (optional): epsilon for LayerNormalization
        """
        super(DecoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist

        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, max_rel_dist, bias)
        self.ffn = FeedForwardNetwork(hidden_dim, ff_hidden_dim, bias)

        self.layernorm1 = nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        """
        Forward pass through decoder layer (the parameter names are for nn.TransformerDecoder).
        Params:
        - torch.Tensor tgt: the input queries tensor from the previous layer
        - torch.Tensor tgt_mask (optional): the mask tensor for input
        """
        # Multi-head attention block
        attn_out = self.layernorm1(tgt)
        attn_out = self.self_attn(attn_out, attn_out, attn_out, mask=tgt_mask)
        attn_out = self.dropout1(attn_out)
        attn_out = tgt + attn_out

        # Feedforward network
        ffn_out = self.layernorm2(attn_out)
        ffn_out = self.ffn(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = ffn_out + attn_out

        return ffn_out


"""
The Music Transformer model.
"""


class MusicTransformer(nn.Module):
    """
    A transformer decoder with relative attention consisting of an input embedding layer, 
    absolute positional encoding, `num_layers` decoder layers, and a final linear layer.
    """

    def __init__(self, hidden_dim, num_layers, num_heads, max_rel_dist, max_abs_position,
                 ff_hidden_dim, vocab_size, bias, dropout, layernorm_eps):
        """
        Params:
        - int hidden_dim: the hidden dimension size of the transformer
        - int ff_hidden_dim: the intermediate dimension size of the feedforward network
        - int num_layers: the number of decoder layers
        - int num_heads: the number of attention heads
        - int max_rel_dist: maximum relative distance between positions to consider in creating relative position embeddings 
        (0 gives normal attention)
        - int max_abs_position: maximum absolute position for which to create a sinusoidal absolute position encoding 
        (0 gives relative attention)
        - int vocab_size: the size of the vocabulary events
        - bool bias (optional): if set to True, all linear layers in the feedforward network will learn an additive bias
        - float dropout (optional): dropout rate for training the model
        - float layernorm_eps (optional): epsilon for LayerNormalization
        """
        super(MusicTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_rel_dist = max_rel_dist
        self.max_abs_position = max_abs_position
        self.ff_hidden_dim = ff_hidden_dim
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = abs_positional_encoding(max_abs_position, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(hidden_dim=hidden_dim, num_heads=num_heads, ff_hidden_dim=ff_hidden_dim, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=hidden_dim, eps=layernorm_eps)
        )

        self.final = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass through the transformer.
        Embed x, add absolute positional encoding (if present), perform dropout, pass through stack of decoder
        layers, and project into vocab space.
        Params:
        - torch.Tensor x: input batch of sequences of shape (batch_size, seq_len)
        - torch.Tensor mask (optional): mask for input batch
        Return:
        - torch.Tensor x: vocabulary event
        """
        # Embed x
        x = self.input_embedding(x)
        x *= math.sqrt(self.hidden_dim)

        # Add absolute positional encoding if max_abs_position > 0, and assuming max_abs_position >> seq_len
        if self.max_abs_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # Dropout
        x = self.input_dropout(x)

        # Pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # Final projection to vocabulary space
        return self.final(x)
