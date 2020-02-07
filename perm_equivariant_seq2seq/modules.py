# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 50


def get_cell_type(cell_type):
    assert cell_type in ['RNN', 'GRU', 'LSTM'], "Please specify layer type as 'GRU' or 'LSTM'"
    if cell_type == 'RNN':
        rnn_model = nn.RNN
    elif cell_type == 'GRU':
        rnn_model = nn.GRU
    elif cell_type == 'LSTM':
        rnn_model = nn.LSTM
    return rnn_model


class EncoderRNN(nn.Module):
    """Standard RNN encoder (using GRU hidden units in recurrent network)

    Args:
        input_size (int): Dimensionality of elements in input sequence number of words in input language)
        semantic_n_hidden (int): Dimensionality of semantic embeddings
        hidden_size (int): Dimensionality of elements in output sequence (number of units in layers)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
        bidirectional (bool, optional) Indicate whether to use a bi-directional encoder (slower). Default: False
        num_layers: (int, optional) Number of layers to place in encoder model. Default: 1
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 layer_type='GRU',
                 semantic_n_hidden=120,
                 bidirectional=False,
                 num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.semantic_n_hidden = semantic_n_hidden
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.semantic_embedding = nn.Embedding(input_size, self.semantic_n_hidden)
        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=self.num_layers)

    def forward(self,
                input,
                hidden):
        """Forward pass through RNN-based encoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, semantic embeddings
        """
        embedded = self.embedding(input.squeeze())[:, None, :]
        semantics = self.semantic_embedding(input.squeeze())
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden, semantics

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device))


class InvariantRNN(nn.Module):
    """RNN encoder with syntax module (with language invariances baked in)

    Args:
        syntax_vocab_size (int): Number of elements in syntax language (accounting for invariances)
        semantic_vocab_size (int): Dimensionality of elements in input sequence number of words in input language)
        hidden_size (int): Dimensionality of elements in output sequence (number of units in layers)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
        semantic_n_hidden (int, optional): Dimensionality of semantic embedding. Default: 120
        bidirectional (bool, optional): Indicate whether to use a bi-directional encoder (slower). Default: False
        num_layers (int, optional): Number of layers to place in encoder model. Default: 1
    """
    def __init__(self,
                 syntax_vocab_size,
                 semantic_vocab_size,
                 hidden_size,
                 layer_type='GRU',
                 semantic_n_hidden=120,
                 bidirectional=False,
                 num_layers=1):
        super(InvariantRNN, self).__init__()
        self.semantic_vocab_size = semantic_vocab_size
        self.syntax_size = syntax_vocab_size
        self.hidden_size = hidden_size
        self.semantic_n_hidden = semantic_n_hidden
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(self.syntax_size, hidden_size)
        self.semantic_embedding = nn.Embedding(self.semantic_vocab_size, self.semantic_n_hidden)

        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=self.num_layers)

    def forward(self,
                input,
                syntax_input,
                hidden):
        """Forward pass through RNN-based encoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            syntax_input (torch.tensor) batch x length x semantic_vocab_size tensor of semantic embeddings
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, semantic embeddings
        """
        syntax_embedded = self.embedding(syntax_input.squeeze())[:, None, :]
        semantics_embedded = self.semantic_embedding(input.squeeze())
        output = syntax_embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden, semantics_embedded

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU):
            return torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device))


class DecoderRNN(nn.Module):
    """Standard RNN decoder (using GRU hidden units in recurrent network)

    Args:
        hidden_size (int): Dimensionality of elements in input sequence
        output_size (int): Dimensionality of elements in output sequence (number of words in output language)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
    """
    def __init__(self,
                 hidden_size,
                 output_size,
                 layer_type='GRU'):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, hidden_size)
        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                input,
                hidden):
        """Forward pass through RNN based decoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, semantic embeddings
        """
        embedded = self.embedding(input).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU) or isinstance(self.rnn, nn.RNN):
            return torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device))


class AttnDecoderRNN(nn.Module):
    """RNN decoder (using GRU hidden units in recurrent network) with attention mechanism on the input sequence

    Args:
        hidden_size (int): Dimensionality of elements in input sequence
        output_size (int): Dimensionality of elements in output sequence (number of words in output language)
        layer_type (str, optional): Specifies type of RNN layer to be used ('GRU' or 'LSTM'). Default: GRU
        dropout_p (float, optional): Dropout rate for embeddings. Default: 0.1
        max_length (int, optional): Maximum allowable length of input sequence (required for attention).
                                    Default: MAX_LENGTH
    """
    def __init__(self,
                 hidden_size,
                 output_size,
                 layer_type='GRU',
                 dropout_p=0.1,
                 max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.rnn_model = get_cell_type(layer_type)
        self.rnn = self.rnn_model(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,
                input,
                hidden,
                encoder_outputs):
        """Forward pass through attention decoder

        Args:
            input (torch.tensor): batch x length x input_size tensor of inputs to decoder
            hidden (torch.tensor): batch x 1 x hidden_size tensor of initial hidden states
            encoder_outputs (torch.tensor): batch x length x hidden_size tensor of encoder hidden states
        Returns:
            (tuple:: torch.tensor, torch.tensor, torch.tensor) decoder output, hidden states, attention weights
        """
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_state = hidden[0] if isinstance(hidden, tuple) else hidden
        attn_weights = F.softmax(attn_state[0] @ encoder_outputs.squeeze().t(), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.permute(1, 0, 2))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        """Initialize the weights of the recurrent model"""
        if isinstance(self.rnn, nn.GRU):
            return torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device)
        elif isinstance(self.rnn, nn.LSTM):
            return (torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device),
                    torch.zeros(self.num_directions * self.num_layers, 1, self.hidden_size, device=device))
