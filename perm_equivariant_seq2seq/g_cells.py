# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import six
import abc

import torch
import torch.nn as nn

from perm_equivariant_seq2seq.g_layers import GroupConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRecurrentCell(six.with_metaclass(abc.ABCMeta, nn.Module)):
    """Abstract base class for group-equivariant recurrent cell layers"""
    def __init__(self,
                 symmetry_group,
                 hidden_size,
                 nonlinearity):
        super(GRecurrentCell, self).__init__()
        self.symmetry_group = symmetry_group
        self.hidden_size = hidden_size
        assert nonlinearity in ['tanh', 'relu'], "Non-linearity not one of 'tanh' or 'relu"
        self.nonlinearity = nonlinearity

    @abc.abstractmethod
    def forward(self,
                input,
                hidden):
        """Forward pass through recurrent cell.

        Args:
            input (torch.tensor): Representation of input group. Rank 2 tensor with dim=(|G|, K)
            hidden (torch.tensor): Representation of hidden inputs. Rank 2 tensor with dim=(|G|, K)
        Returns (
            (torch.tensor) Hidden state representation. Rank 2 tensor with dim=(|G|, K)
        """

    def init_hidden(self):
        return torch.zeros(1, self.symmetry_group.size, self.hidden_size, device=device)


class GRNNCell(GRecurrentCell):
    """Implementation of a simple group equivariant RNN cell. Cell has the following form:

        GRNN(word, hidden):
            Input:  word:   Z -> Rk
                    hidden: G -> Rk
            Output: h_out:  G -> Rk

            f = [word * psi_f] (g);    for every g in G
            h = [hidden * psi_h] (g);  for every g in G
            h_out = sigma( f + h )

    Args:
        symmetry_group (Symmetry): object representing the symmetry group layer should be equivariant to
        hidden_size (int): dimensionality of the output filters
        nonlinearity (str, optional): tanh or relu for activation function of the cell. Default: 'tanh'
    """
    def __init__(self,
                 symmetry_group,
                 hidden_size,
                 nonlinearity='tanh'):
        super(GRNNCell, self).__init__(symmetry_group, hidden_size, nonlinearity)

        # Initialize input token convolution
        self.psi_w = GroupConv(symmetry_group=self.symmetry_group,
                               input_size=self.hidden_size,
                               embedding_size=self.hidden_size)

        # Initialize hidden state convolution
        self.psi_h = GroupConv(symmetry_group=self.symmetry_group,
                               input_size=self.hidden_size,
                               embedding_size=self.hidden_size)

        # Define non-linearity
        self.activation = nn.Tanh() if nonlinearity == 'tanh' else nn.ReLU()

    def forward(self, input, hidden):
        """Compute a forward pass through the RNN cell

        Args:
            input (torch.tensor): 1-hot representation of word at time t
            hidden (torch.tensor): hidden state at time t (of form G-->R^k, i.e., R^|G| x R^k)
        Returns
            (torch.tensor) output of RNN cell
        """
        f = self.psi_w(input)
        h = self.psi_h(hidden)
        return self.activation(f + h)


class GLSTMCell(GRecurrentCell):
    """Implementation of a group equivariant LSTM cell.

    Args:
        symmetry_group (Symmetry): object representing the symmetry group layer should be equivariant to
        hidden_size (int): dimensionality of the output filters
    """
    def __init__(self,
                 symmetry_group,
                 hidden_size):
        super(GLSTMCell, self).__init__(symmetry_group, hidden_size, nonlinearity='tanh')
        # Initialize convolutional layers for it
        self.psi_ii = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hi = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize convolutional layers for ft
        self.psi_if = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hf = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize convolutional layers for gt
        self.psi_ig = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hg = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize convolutional layers for ot
        self.psi_io = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_ho = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize non-linearities
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        """Compute a forward pass through the LSTM cell

        Args:
            input (torch.tensor): 1-hot representation of word at time t
            hidden (torch.tensor): hidden state at time t (of form G-->R^k, i.e., R^|G| x R^k)
        Returns
            (torch.tensor) output of RNN cell
        """
        h, c = hidden
        i = self.sigmoid(self.psi_ii(input) + self.psi_hi(h))
        f = self.sigmoid(self.psi_if(input) + self.psi_hf(h))
        g = self.tanh(self.psi_ig(input) + self.psi_hg(h))
        o = self.sigmoid(self.psi_io(input) + self.psi_ho(h))
        c = f * c + i * g
        h = o * self.tanh(c)
        return h, c

    def init_hidden(self):
        return tuple([super(GLSTMCell, self).init_hidden() for _ in range(2)])


class GGRUCell(GRecurrentCell):
    """Implementation of a group equivariant GRU cell.

    Args:
        symmetry_group (Symmetry): object representing the symmetry group layer should be equivariant to
        hidden_size (int): dimensionality of the output filters
    """
    def __init__(self,
                 symmetry_group,
                 hidden_size):
        super(GGRUCell, self).__init__(symmetry_group, hidden_size, nonlinearity='tanh')

        # Initialize convolutional layers for rt
        self.psi_ir = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hr = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize convolutional layers for zt
        self.psi_iz = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hz = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize convolutional layers for nt
        self.psi_in = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        self.psi_hn = GroupConv(symmetry_group=self.symmetry_group,
                                input_size=self.hidden_size,
                                embedding_size=self.hidden_size)
        # Initialize non-linearities
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        """Compute a forward pass through the GRU cell

        Args:
            input (torch.tensor): 1-hot representation of word at time t
            hidden (torch.tensor): hidden state at time t (of form G-->R^k, i.e., R^|G| x R^k)
        Returns
            (torch.tensor) output of RNN cell
        """
        r = self.sigmoid(self.psi_ir(input) + self.psi_hr(hidden))
        z = self.sigmoid(self.psi_iz(input) + self.psi_hz(hidden))
        n = self.tanh(self.psi_in(input) + r * self.psi_hn(hidden))
        return (1 - z) * n + z * hidden
