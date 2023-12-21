import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import unittest

def attention_matrix(q, k, dropout=None, mask=None):
    """
    Compute the attention matrix given Query, Key, and Value tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        dropout (nn.Dropout, optional): Dropout layer for regularization. Default is None.
        mask (torch.Tensor, optional): Mask for attention scores. Default is None.

    Returns:
        torch.Tensor: Attention matrix.
    """
    d_k = q.size(-1) 
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k) 
 

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Set a very high negative attention score for masked entries.

    att_matrix = scores.softmax(dim=-1)
   
    if dropout is not None:
        att_matrix = dropout(att_matrix)  # Apply dropout during training 

    return att_matrix


def attention(q, k, v, dropout=None, mask=None):
    """
    Compute the attention-based output given Query, Key, and Value tensors.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        dropout (nn.Dropout, optional): Dropout layer for regularization. Default is None.
        mask (torch.Tensor, optional): Mask for attention scores. Default is None.

    Returns:
        torch.Tensor: Attention-based output.
    """

    att_matrix = attention_matrix(q, k)
    return torch.matmul(att_matrix, v), att_matrix


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.1, trainable=True):
        """
        Initialize a Multihead Attention layer.

        Args:
            input_size (int): Input size of the layer.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability. Default is 0.1.
            trainable (bool, optional): If True, parameters are trainable. Default is True.
        """
        super(MultiheadAttentionLayer, self).__init__()

        assert input_size % num_heads == 0, "Input size must be divisible by the number of heads."

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = input_size // num_heads
        self.attention_matrix_list = []
        self.attention_based_v_list = []
        self.attention_matrix = None
        self.attention_based_v = None

        # Linear projections for Query, Key, and Value
        self.W_q = nn.Linear(input_size, input_size, bias=False)
        self.W_k = nn.Linear(input_size, input_size, bias=False)
        self.W_v = nn.Linear(input_size, input_size, bias=False)

        # Output projection
        self.W_o = nn.Linear(input_size, input_size, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Set requires_grad based on the trainable parameter
        for param in self.parameters():
            param.requires_grad = trainable

        # Parameter initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)


    def forward(self, x):
        """
        Forward pass of the Multihead Attention layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Linear projections
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Split into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_size).transpose(1, 2)

        # Scaled Dot-Product Attention
        self.attention_based_v, self.attention_matrix = attention(q, k, v)
        #print(self.attention_matrix)

        # Concatenate and project back to the original size
        self.attention_based_v = self.attention_based_v.transpose(1, 2).contiguous().view(x.size(0), -1, self.input_size)
        if self.num_heads > 1:
            self.attention_based_v = self.W_o(self.attention_based_v).squeeze(dim=1)
        self.attention_matrix = self.attention_matrix.squeeze()
        self.attention_matrix_list.append(self.attention_matrix)
        self.attention_based_v_list.append(self.attention_based_v)

        return self.attention_based_v


class TestMultiheadAttention(unittest.TestCase):

    def test_forward(self):
        # Define input tensor
        word_len = 4
        num_heads = 2
        batch_size = 2
        sequence_length = 2

        # Create a MultiheadAttention layer
        attention_layer = MultiheadAttentionLayer(input_size=word_len, num_heads=num_heads)

        # Generate random input tensor
        input_tensor = torch.randn(batch_size, sequence_length, word_len)
        #print(input_tensor)

        # Forward pass through the attention layer
        output_tensor = attention_layer(input_tensor)
        #print(output_tensor)

        # Ensure the output tensor has the correct shape
        expected_shape = torch.Size([batch_size, word_len])
        self.assertEqual(output_tensor.shape, input_tensor.shape)

if __name__ == '__main__':
    # Run the tests
    unittest.main()