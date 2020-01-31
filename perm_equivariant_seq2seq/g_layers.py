import math

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordConv(nn.Module):
    """Convolution of a word w with a filter psi: Z->Rk.

        Input:  w:   Z -> {0,1}
                psi: Z -> Rk
        Output: phi: G -> Rk

    Convolution has the form:

            phi: [w * psi](g) = sum_Z w(z) psi(g^{-1} z)
                              = psi(g^{-1} z)             ; for every g in G

    Note that w(z) = 1 at the word, and 0 everywhere else. G is the group w.r.t. which we want to be equivariant.

    Args:
        symmetry_group (PermutationSymmetry): Object representing the symmetry group
        vocabulary_size (int): Number of words in language
        embedding_size (int): Dimensionality of filters psi (k)
    """
    def __init__(self,
                 symmetry_group,
                 vocabulary_size,
                 embedding_size):
        super(WordConv, self).__init__()
        self.symmetry_group = symmetry_group
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

    def forward(self, input):
        """Forward pass through the layer

        Args:
        input (torch.tensor): w:Z -> {0, 1} 1-hot representation of the word to be convolved (batch_size x V)
        Returns:
            (torch.tensor) f:G -> Rk functional output over the group (batch_size x |G| x K)
        """
        word_permutations = self.permute_word(input)
        permutation_indices = torch.argmax(word_permutations, dim=-1)
        return self.embedding(permutation_indices).squeeze()

    def permute_word(self, word):
        return torch.stack([word @ self.symmetry_group.index2inverse[i]
                            for i in range(self.symmetry_group.size)], dim=1)


class GroupClassifier(WordConv):
    """Convolution of two group representation (f and psi) evaluated at a series of integer locations

            f:     G -> Rk
            psi:   G -> Rk

    Convolution has the form:

           phi: [f * psi](z) = sum_{g in G} sum_{k in K} f(g) psi_k(g^{-1} z)

    Therefore, the output has the form f: Z -> R, which can then be used for classification by using e.g., a SoftMax

    Args:
        symmetry_group (PermutationSymmetry): Object representing the symmetry group
        vocabulary_size (int): Number of words in language
        embedding_size (int): Dimensionality of filters psi (k)
    """

    def __init__(self,
                 symmetry_group,
                 vocabulary_size,
                 embedding_size):
        super(GroupClassifier, self).__init__(symmetry_group=symmetry_group,
                                              vocabulary_size=vocabulary_size,
                                              embedding_size=embedding_size)
        self.bias = torch.nn.Parameter(1e-1 * torch.ones(self.vocabulary_size))
        self._init_parameters()
        # Pre-compute permutations
        words = torch.eye(self.vocabulary_size, dtype=torch.float64).to(device)
        word_permutations = [torch.argmax(self.permute_word(word.unsqueeze(0)), dim=1) for word in words]
        self.word_permutations = torch.nn.Parameter(torch.stack(word_permutations), requires_grad=False)

    def forward(self, input):
        """Forward pass through the layer to yield the classification logits

        Args:
            input (torch.tensor): f: G -> R^k matrix representation of group to classify  (batch_size x |G| x K)
        Returns:
            (torch.tensor) Z -> R representing classification logits                    (batch_size x  1  x V)
        """
        batch_size = input.shape[0]
        ipt = input.view(batch_size, -1)                                    # dim: batch x (|G| * K)
        conv = self.conv_filter().view(self.vocabulary_size, -1)            # dim: V x (|G| * K)
        logits = ipt @ conv.t() + self.bias                                 # dim: batch x V
        return logits.unsqueeze(1)                                          # dim: batch x 1 x V

    def conv_filter(self):
        return self.embedding(self.word_permutations)

    def _init_parameters(self):
        """Use a standard uniform initialization for the parameters of the module"""
        stdv = 1.0 / math.sqrt(self.embedding_size)
        torch.nn.init.uniform_(self.embedding.weight, -stdv, stdv)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.embedding.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def permute_word(self, word):
        """Helper function to return all permutations of a word under the symmetry group"""
        return torch.cat([word @ self.symmetry_group.index2inverse[h]
                          for h in range(self.symmetry_group.size)], dim=0)


class GroupConv(nn.Module):
    """Convolution of two functions (f and psi) on a group G.

            f:     G -> Rk
            psi^i: G -> Rk'
            psi:   G -> Rk x Rk'

    Convolution has the form:

           phi: [f * psi^i](g) = sum_{h in G} sum_{k in K} f(h) psi^i_k(g^{-1} h);    for every g in G, i=1,...,K'

    Therefore, the output has the form f: G -> Rk'

    Args:
        symmetry_group (PermutationSymmetry): Object representing the symmetry group
        input_size (int): Dimensionality of input representation (k)
        embedding_size (int): Dimensionality of filters psi (k')
    """

    def __init__(self,
                 symmetry_group,
                 input_size,
                 embedding_size):
        super(GroupConv, self).__init__()
        self.symmetry_group = symmetry_group
        self.input_size = input_size
        self.embedding_size = embedding_size

        # Initialize filters
        self.weight = torch.nn.Parameter(torch.Tensor(self.symmetry_group.size, self.input_size, self.embedding_size))
        self.bias = torch.nn.Parameter(1e-1 * torch.ones(self.embedding_size))
        self._init_parameters()

    def forward(self, input):
        """Forward pass through the layer

        Args:
            input (torch.tensor): f: G -> Rk, represented as |G| x Rk tensor   (batch_size x |G| x K)
        Returns:
            (torch.tensor) phi: G -> Rk', represented as a |G| x Rk' tensor  (batch_size x |G| x K)
        """
        ipt = input[:, None, ..., None]                           # dim: batch x      1   x |G| (h) x  K  x 1
        conv_filter = self.get_conv_filter()[None, ...]           # dim:   1   x  |G| (g) x |G| (h) x  K  x K
        return (ipt * conv_filter).sum(2).sum(2) + self.bias

    def get_conv_filter(self):
        return torch.stack([torch.index_select(self.weight, 0, self.symmetry_group.index2inverse_indices[g])
                            for g in range(self.symmetry_group.size)])

    def _init_parameters(self):
        """Helper function to return all permutations of a word under the symmetry group"""
        stdv = 1.0 / math.sqrt(self.embedding_size)
        torch.nn.init.uniform_(self.weight, -stdv, stdv)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)
