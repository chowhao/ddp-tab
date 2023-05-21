import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

# tab transformer model of pytorch
model = TabTransformer(
    categories=(2500, 2500, 2500),
    num_continuous=1,
    dim=32,  # dimension, paper set at 32
    dim_out=1,
    depth=6,  # depth, paper recommended 6
    heads=8,  # heads, paper recommends 8
    attn_dropout=0.1,  # post-attention dropout
    ff_dropout=0.1,  # feed forward dropout
    # ff_dropout=0.4,  # feed forward dropout
    mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension
    mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu
)
