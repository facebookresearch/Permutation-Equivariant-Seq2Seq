# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

from perm_equivariant_seq2seq.g_layers import WordConv, GroupConv, GroupClassifier
from perm_equivariant_seq2seq.g_layers import LSGroupClassifier, LSGRoupEmbedding
from perm_equivariant_seq2seq.g_cells import GRNNCell, GGRUCell, GLSTMCell
from perm_equivariant_seq2seq.symmetry_groups import LearnablePermutationSymmetry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EquivariantDecoderRNN(nn.Module):
    """Equivariant implementation of a decoder RNN, that maps word tokens and equivariant hidden states to word tokens.

    Args:
        hidden_size (int): Number of hidden units in RNN model
        output_size (int): Size of output vocabulary
        symmetry_group (PermutationSymmetry): Object representing the symmetry group
        cell_type (str, optional): Type of recurrent layer to use (currently only standard GRNN supported).
                                   Default: 'GRNN'
        nonlinearity (str, optional): Nonlinear activation to use in RNN cell (tanh or relu). Default: 'tanh'
    """
    def __init__(self,
                 hidden_size,
                 output_size,
                 symmetry_group,
                 cell_type='GRNN',
                 nonlinearity='tanh'):
        super(EquivariantDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.symmetry_group = symmetry_group
        self.nonlinearity = nonlinearity

        # Initialize word convolution embedding
        self.embedding = WordConv(symmetry_group=self.symmetry_group,
                                  vocabulary_size=self.output_size,
                                  embedding_size=self.hidden_size)
        # Initialize G-recurrent cell
        assert cell_type in ['GRNN', 'GGRU'], "Currently only GRNN GGRU supported"
        if cell_type == 'GRNN':
            self.recurrent_cell = GRNNCell(symmetry_group=self.symmetry_group,
                                           hidden_size=self.hidden_size,
                                           nonlinearity=self.nonlinearity)
        elif cell_type == 'GGRU':
            self.recurrent_cell = GGRUCell(symmetry_group=self.symmetry_group,
                                           hidden_size=self.hidden_size)

        # Initialize linear layer and softmax for output
        self.out = GroupClassifier(symmetry_group=self.symmetry_group,
                                   vocabulary_size=self.output_size,
                                   embedding_size=self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                input,
                hidden):
        """Forward pass through decoder RNN.

        Args:
            input (torch.tensor): w:Z -> {0, 1} 1-hot representation of the previous word (batch_size x V)
            hidden (torch.tensor): f: G -> Rk, represented as |G| x Rk tensor   (batch_size x |G| x K)
        Returns:
            (tuple::torch.tensor, torch.tensor) output, hidden-state

        """
        embedded = self.embedding(input).unsqueeze(0)
        hidden = self.recurrent_cell(embedded, hidden)
        output = self.softmax(self.out(hidden).squeeze(0))
        return output, hidden


class EquivariantAttnDecoder(nn.Module):
    """Equivariant implementation of an attention decoder RNN

    Args:
        hidden_size (int) Number of hidden units in RNN model
        output_size (int) Size of output vocabulary
        symmetry_group (PermutationSymmetry): Object representing the symmetry group
        cell_type (str, optional) Type of recurrent layer to use. Default: GRNN
        nonlinearity (str, optional) Nonlinear activation to use in RNN cell (tanh or relu). Default: tanh
    """
    def __init__(self,
                 hidden_size,
                 output_size,
                 symmetry_group,
                 cell_type='GRNN',
                 nonlinearity='tanh'):
        super(EquivariantAttnDecoder, self).__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.symmetry_group = symmetry_group
        self.nonlinearity = nonlinearity
        self.activation = nn.Tanh() if self.nonlinearity == 'tanh' else nn.ReLU()

        # Initialize word convolution embedding
        self.embedding = WordConv(symmetry_group=self.symmetry_group,
                                  vocabulary_size=self.output_size,
                                  embedding_size=self.hidden_size)

        # Initialize attention mixing layers
        self.attn_combine = GroupConv(symmetry_group=self.symmetry_group,
                                      input_size=self.hidden_size * 2,
                                      embedding_size=self.hidden_size)
        # Initialize G-recurrent cell
        assert self.cell_type in ['GRNN', 'GGRU', 'GLSTM'], "Supported cell types: GRNN, GGRU, and GLSTM"
        if self.cell_type == 'GRNN':
            self.recurrent_cell = GRNNCell(symmetry_group=self.symmetry_group,
                                           hidden_size=self.hidden_size,
                                           nonlinearity=self.nonlinearity)
        elif self.cell_type == 'GGRU':
            self.recurrent_cell = GGRUCell(symmetry_group=self.symmetry_group,
                                           hidden_size=self.hidden_size)
        elif self.cell_type == 'GLSTM':
            self.recurrent_cell = GLSTMCell(symmetry_group=self.symmetry_group,
                                            hidden_size=self.hidden_size)
        # Initialize linear layer and softmax for output
        classifier = LSGroupClassifier if self.learnable else GroupClassifier
        self.out = classifier(symmetry_group=self.symmetry_group,
                              vocabulary_size=self.output_size,
                              embedding_size=self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                input,
                hidden,
                encoder_outputs):
        """Forward pass through attention decoder (with dot-product attention implemented)

        Args:
            input: (torch.tensor) Input word at time t             w: Z -> {0, 1} (one-hot encoding)
            hidden: (torch.tensor) Hidden state at time t-1        s: G --> R^K   (rank 3 tensor |G| x 1 x K)
            encoder_outputs: (torch.tensor) Encoder hidden states                 (rank 3 tensor T x | G| x K)
        Returns:
            (tuple::torch.tensor, torch.tensor, torch.tensor) output, hidden-state, attention weights
        """
        embedded = self.embedding(input).unsqueeze(0)

        # Compute attention weights
        attn_state = hidden[0] if self.cell_type == 'GLSTM' else hidden
        hidden_flat, encoder_flat = attn_state.view(1, -1), encoder_outputs.view(encoder_outputs.shape[0], -1)
        attn_weights = F.softmax(hidden_flat @ encoder_flat.t(), dim=1).t()
        attn_applied = (attn_weights.unsqueeze(2) * encoder_outputs).sum(0, keepdim=True)
        # Mix attention and embedded
        output = torch.cat((embedded, attn_applied), dim=-1)
        output = torch.tanh(self.attn_combine(output))
        # Compute recurrent
        hidden = self.recurrent_cell(output, hidden)
        output = hidden[0] if isinstance(hidden, tuple) else hidden
        output = self.softmax(self.out(output).squeeze(0))
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros((1, self.symmetry_group.size, self.hidden_size)).to(device)


class EquiRNN(nn.Module):
    """Wrapper class for RNN-cells, mainly useful for encoders that can encode complete sequences

    Args:
        symmetry_group (SymmetryGroup): Object representing the symmetry group w.r.t. model is equivariant
        input_size (int): Size of input vocabulary
        hidden_size (int): Size of latent representation
        cell_type (str): Type of G-Recurrent cell to use
        bidirectional (bool, optional): Whether to use bidirectional RNN or not. Default: False
        nonlinearity (str, optional) Activation function to be used in G-Recurrent cell (tanh or relu). Defalt: tanh
    """
    def __init__(self,
                 symmetry_group,
                 input_size,
                 hidden_size,
                 cell_type,
                 bidirectional=False,
                 nonlinearity='tanh'):
        super(EquiRNN, self).__init__()
        self.symmetry_group = symmetry_group
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        assert nonlinearity in ['tanh', 'relu'], "Non-linearity not one of 'tanh' or 'relu"
        self.nonlinearity = nonlinearity

        # Initialize word convolution embedding
        self.embedding = WordConv(symmetry_group=self.symmetry_group,
                                  vocabulary_size=self.input_size,
                                  embedding_size=self.hidden_size)
        # Initialize G-recurrent cell
        if self.cell_type == 'GRNN':
            self.cell = GRNNCell(symmetry_group=self.symmetry_group,
                                 hidden_size=self.hidden_size,
                                 nonlinearity=self.nonlinearity)
        elif self.cell_type == 'GGRU':
            self.cell = GGRUCell(symmetry_group=self.symmetry_group,
                                 hidden_size=self.hidden_size)
        elif self.cell_type == 'GLSTM':
            self.cell = GLSTMCell(symmetry_group=self.symmetry_group,
                                  hidden_size=self.hidden_size)
        else:
            raise NotImplementedError('Currently only supports GRNN, GGRU, or GLSTM cell types')

    def forward(self, sequence, h_0=None):
        """Forward pass through the model

        Args:
            sequence (torch.tensor): 1-hot representation of sequence  (batch_size x T x V)
            h_0 (torch.tensor, optional): Initial hidden state (batch_size x |G| x K). Default: None
        Returns:
            (tuple: torch.tensor, torch.tensor): hidden-states, final hidden-state
        """
        embedded_sequence = self.embedding(sequence.squeeze())
        hidden_all, ht = self.forward_pass(embedded_sequence, h_0)
        if self.bidirectional:
            reverse_sequence = torch.cat((embedded_sequence[:-1].flip(0), embedded_sequence[-1][None, :]))
            bwd_h, ht_b = self.forward_pass(reverse_sequence, h_0)
            bwd_h = torch.cat((bwd_h[:-1].flip(0), bwd_h[-1][None, ...]))
            hidden_all = torch.cat((hidden_all, bwd_h), dim=-1)
            if self.cell_type == 'GRNN' or self.cell_type == 'GGRU':
                ht = torch.cat((ht, ht_b), dim=-1)
            elif self.cell_type == 'GLSTM':
                ht = tuple([torch.cat((h_f, h_b), dim=-1) for h_f, h_b in zip(ht, ht_b)])
        return hidden_all, ht

    def forward_pass(self, sequence, h_0=None):
        """Forward pass through the model

        Args:
            sequence (torch.tensor): 1-hot representation of sequence  (batch_size x T x V)
            h_0 (torch.tensor, optional): Initial hidden state (batch_size x |G| x K). Default: None
        Returns:
            (tuple: torch.tensor, torch.tensor): hidden-states, final hidden-state
        """
        h = self.init_hidden() if h_0 is None else h_0
        seq_length = sequence.shape[0]
        hidden_all = torch.zeros(seq_length, self.symmetry_group.size, self.hidden_size).to(device)
        for t, element in enumerate(sequence):
            h = self.cell(element[None, ...], h)
            state = h[0] if self.cell_type == 'GLSTM' else h
            hidden_all[t] = state.squeeze()
        return hidden_all, h

    def init_hidden(self):
        return self.cell.init_hidden()
