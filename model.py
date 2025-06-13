import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Input embedding layer that converts token indices to dense vectors.
    Args:
        d_model (int): Dimension of the model
        vocab_size (int): Size of the vocabulary
    """
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds positional information to input embeddings.
    Args:
        d_model (int): Dimension of the model
        seq_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    def __init__(self , d_model : int , seq_len : int , dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create position vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Create division term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Apply sinusoidal encoding to even and odd indices
        pe[: , 0::2] = torch.sin(position * div_term)  # even indices
        pe[: , 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, seq_len, d_model)
        self.register_buffer('pe', pe) # register the pe matrix as a buffer to the model not as a parameter
    
    def forward(self , x):
        x = x + (self.pe[: , :x.shape[1] , : ]).requires_grad_(False) # add the positional encoding to the input
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization and dropout.
    Args:
        features (int): Number of features in the input
        dropout (float): Dropout probability
    """
    def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
    def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class LayerNormalization(nn.Module):
    """
    Layer normalization with learnable parameters.
    Args:
        features (int): Number of features in the input
        eps (float): Small value to prevent division by zero
    """
    def __init__(self , features : int , eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable shift parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        mean = x.mean(dim = -1, keepdim = True)  # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True)    # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Feed-forward network with two linear layers and ReLU activation.
    Args:
        d_model (int): Input and output dimension
        d_ff (int): Hidden dimension of the feed-forward network
        dropout (float): Dropout probability
    """
    def __init__(self , d_model : int , d_ff : int , dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model , d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff , d_model)

    def forward(self , x):
        # x : (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention mechanism.
    Args:
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    def __init__(self , d_model : int , n_heads : int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads  # Dimension of each head's key/query/value vectors
        
        # Linear projections for query, key and value
        self.linear_q = nn.Linear(d_model, d_model , bias = False)  # Query projection
        self.linear_k = nn.Linear(d_model, d_model , bias = False)  # Key projection
        self.linear_v = nn.Linear(d_model, d_model , bias = False)  # Value projection
        
        self.linear_out = nn.Linear(d_model, d_model , bias = False)  # Output projection
        self.dropout = nn.Dropout(dropout)  # Dropout for regularization

    @staticmethod
    def attention(query, key, value, mask , dropout : nn.Dropout):
            d_k = query.shape[-1]
              # Just apply formula from paper
            attention_scores = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)  # (batch, n_heads, seq_len, seq_len)
            if mask is not None:
                     # Write a very low value (indicating -inf) to the positions where mask == 0
                attention_scores = attention_scores.masked_fill(mask == 0 , -1e9)
                attention_scores = torch.softmax(attention_scores , dim = -1)
            if dropout is not None:
                   attention_scores = dropout(attention_scores)
            return (attention_scores @ value) , attention_scores

    
    def forward(self , q, v, k , mask):

        query = self.linear_q(q)  # (batch, seq_len, d_model)
        key = self.linear_k(k)    # (batch, seq_len, d_model)
        value = self.linear_v(v)  # (batch, seq_len, d_model)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.n_heads, self.d_k).transpose(1, 2)

        x , self.attention_scores = self.attention(query , key , value , mask , self.dropout)
        # (batch, n_heads, seq_len, d_k) --> (batch, seq_len, n_heads * d_k)
          
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k)

        return self.linear_out(x)  # (batch, seq_len, d_model)


class EncoderBlock(nn.Module):
    """
    Encoder block consisting of multi-head attention and feed-forward network.
    Args:
        d_model (int): Dimension of the model
        n_heads (int): Number of attention heads
        d_ff (int): Hidden dimension of the feed-forward network
        dropout (float): Dropout probability
    """
    def __init__(self , features : int , self_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features , dropout) for _ in range(2)])

    def forward(self , x , src_mask):
        x = self.residual_connections[0](x , lambda x: self.self_attention_block(x , x , x , src_mask))
        x = self.residual_connections[1](x , self.feed_forward_block)
        return x

class Encoder(nn.Module):
    
    def __init__(self , features : int , layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self , x , mask):
        for layer in self.layers:
            x = layer(x , mask)
        return self.norm(x)
        


# ======== DECODER ========

class DecoderBlock(nn.Module):

    def __init__(self , self_attention_block : MultiHeadAttentionBlock , cross_attention_block : MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , features : int , dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features , dropout) for _ in range(3)])

    def forward(self , x, enc_output , src_mask , tgt_mask):
        x = self.residual_connections[0](x , lambda x : self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x , lambda x : self.cross_attention_block(x , enc_output , enc_output , src_mask))
        x = self.residual_connections[2](x , self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self , features : int , layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self , x , enc_output , src_mask , tgt_mask):
        for layer in self.layers:
            x = layer(x , enc_output , src_mask , tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self , vocab_size : int , d_model : int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocab_size)
    
    def forward(self , x) -> None:
        # x : (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x) , dim = -1)


class Transformer(nn.Module):

    def __init__(self , encoder : Encoder , decoder : Decoder , input_embedding : InputEmbedding , tgt_emedding : InputEmbedding , src_encoding : PositionalEncoding , tgt_encoding : PositionalEncoding ,  projection_layer : ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.tgt_embedding = tgt_emedding
        self.src_encoding = src_encoding
        self.tgt_encoding = tgt_encoding
        self.projection_layer = projection_layer

    def encode(self , src , src_mask):
        x = self.input_embedding(src)
        x = self.src_encoding(x)
        return self.encoder(x , src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size : int , tgt_vocab_size : int , src_seq_len : int , tgt_seq_len : int ,h : int = 8 ,  d_model : int = 512 , N : int = 6 , dropout : float = 0.1 , d_ff : int = 2048) -> Transformer :

    # Input Embedding
    input_embedding = InputEmbedding(d_model , src_vocab_size)
    tgt_embedding = InputEmbedding(d_model , tgt_vocab_size)

    # Positional Encoding
    src_encoding = PositionalEncoding(d_model , src_seq_len , dropout)
    tgt_encoding = PositionalEncoding(d_model , tgt_seq_len , dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout)
        encoder_blocks.append(EncoderBlock(d_model , self_attention_block , feed_forward_block , dropout))

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model , h , dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout)
        decoder_blocks.append(DecoderBlock(self_attention_block , cross_attention_block , feed_forward_block , d_model , dropout))

    # Encoder and Decoder
    encoder = Encoder(d_model , nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model , nn.ModuleList(decoder_blocks))

    # Projection Layer
    projection_layer = ProjectionLayer( d_model , tgt_vocab_size)

    # create transformer
    transformer = Transformer(encoder , decoder , input_embedding , tgt_embedding , src_encoding , tgt_encoding , projection_layer)

    # Initialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    return transformer
