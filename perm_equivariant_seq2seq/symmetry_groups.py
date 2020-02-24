# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageInvariance:
    """Object to represent a language invariance

    Args:
        invariant_set (list): List of strings representing invariant words
        invariant_token (string): String representation of invariant set elements
    """
    def __init__(self, invariant_set, invariant_token):
        super(LanguageInvariance, self).__init__()
        self.invariant_set = invariant_set
        self.invariant_token = invariant_token

    def map_word(self, word):
        return self.invariant_token if word in self.invariant_set else word


def get_permutation_matrix(num_letters, permutations):
    """Construct a permutation matrix (NxN matrix with det(A)=1) that implements that desired permutations

    Args:
        num_letters (int): Number of letters in space (i.e., N)
        permutations (list::tuple): List of tuples of the form (i,j) indicating permuting i-->j
    Returns:
        (np.array) Permutation matrix as specified
    """
    perm_mat = np.zeros((num_letters, num_letters))
    for perm in permutations:
        perm_mat[perm[1] - 1, perm[0] - 1] = 1.
    # Enforce that every row / column has at least one element = 1
    row_sums, columns_sums = perm_mat.sum(1), perm_mat.sum(0)
    for j in range(num_letters):
        if row_sums[j] == 0:
            perm_mat[j, j] = 1.
        if columns_sums[j] == 0:
            perm_mat[j, j] = 1.
    assert np.linalg.det(perm_mat) in [1., -1], "Determinant != -+1., please ensure valid permutations"
    return torch.tensor(perm_mat).to(device)


class PermutationSymmetry:
    """
    Abstract class for groups of permutation symmetries
    """
    def __init__(self, num_letters):
        self.num_letters = num_letters

        # Define identity element for the group
        self.e = torch.eye(self.num_letters, dtype=torch.float64).to(device)
        self.perm_matrices = [self.e]

    def in_group(self, perm_matrix):
        return np.sum([torch.allclose(perm_matrix, mat) for mat in self.perm_matrices]) > 0

    @property
    def size(self):
        return len(self.perm_matrices)

    def mat2index(self, mat):
        """Get the group index of a matrix if it is in the group

        Args:
            mat: (torch.tensor) Tensor representation of matrix
        Returns:
            (int) Index of matrix in group, or None if matrix is not in group
        """
        if not self.in_group(mat):
            return None
        return [i for i, perm_mat in enumerate(self.perm_matrices) if torch.allclose(mat, perm_mat)][0]

    @property
    def learnable(self):
        return False


class CircularShift(PermutationSymmetry):
    """Class to represent a circular shift for some number of letters (must be less than N)

    Args:
        num_letters (int): Number of characters / letters in the vocabulary
        num_equivariant (int): Number of characters in vocabulary that are equivariant (assumed to be in sequence)
        first_equivariant (int, optional): Index of first equivariant character
    """
    def __init__(self, num_letters, num_equivariant, first_equivariant=0):
        super(CircularShift, self).__init__(num_letters)
        self.num_equivariant = num_equivariant
        self.first_equivariant = first_equivariant
        self.last_equivariant = first_equivariant + num_equivariant

        # Define initial shift
        self.init_perm = [(i, i + 1) for i in range(self.first_equivariant, self.last_equivariant - 1)]
        self.init_perm += [(self.last_equivariant - 1, self.first_equivariant)]
        self.tau1 = get_permutation_matrix(self.num_letters, self.init_perm)
        self.perm_matrices.append(self.tau1)
        for _ in range(self.num_equivariant - 2):
            perm_mat = self.perm_matrices[-1] @ self.tau1
            self.perm_matrices.append(perm_mat)

        self.index2mat, self.index2inverse, self.index2inverse_indices = {}, {}, {}
        for idx, mat in enumerate(self.perm_matrices):
            self.index2mat[idx] = mat
            self.index2inverse[idx] = torch.pinverse(mat)
            self.index2inverse_indices[idx] = torch.tensor(
                [self.mat2index(torch.pinverse(mat) @ h) for h in self.perm_matrices],
                dtype=torch.long
            ).to(device)


class VerbDirectionSCAN(PermutationSymmetry):
    """Specific class to capture both verb and direction equivariances in SCAN. Assumes verbs first, then directions"""
    def __init__(self, num_letters, first_equivariant):
        super(VerbDirectionSCAN, self).__init__(num_letters)
        self.num_equivariant = [4, 2]
        self.first_equivariant = [first_equivariant, first_equivariant + self.num_equivariant[0]]
        self.last_equivariant = [fe + ne for fe, ne in zip(self.num_equivariant, self.first_equivariant)]

        # Initialize separate symmetry groups for verbs and directions
        self.verb_sym = CircularShift(self.num_letters, self.num_equivariant[0], self.first_equivariant[0])
        self.dir_sym = CircularShift(self.num_letters, self.num_equivariant[1], self.first_equivariant[1])

        # Construct perm matrices as cartesian product between the groups
        for dir_mat in self.dir_sym.perm_matrices:
            for verb_mat in self.verb_sym.perm_matrices:
                new_mat = dir_mat @ verb_mat
                if not self.in_group(new_mat):
                    self.perm_matrices.append(new_mat)

        self.index2mat, self.index2inverse, self.index2inverse_indices = {}, {}, {}
        for idx, mat in enumerate(self.perm_matrices):
            self.index2mat[idx] = mat
            self.index2inverse[idx] = torch.pinverse(mat)
            self.index2inverse_indices[idx] = torch.tensor(
                [self.mat2index(torch.pinverse(mat) @ h) for h in self.perm_matrices],
                dtype=torch.long
            ).to(device)


class TrivialGroup(PermutationSymmetry):
    """Represent the trivial group G = {e} for testing purposes"""
    def __init__(self, num_letters):
        super(TrivialGroup, self).__init__(num_letters=num_letters)
        self.tau = torch.eye(num_letters, dtype=torch.float64).to(device)
        self.perm_matrices = [self.tau]
        self.index2mat, self.index2inverse, self.index2inverse_indices = {}, {}, {}
        for idx, mat in enumerate(self.perm_matrices):
            self.index2mat[idx] = mat
            self.index2inverse[idx] = torch.pinverse(mat)
            self.index2inverse_indices[idx] = torch.tensor(
                [self.mat2index(torch.pinverse(mat) @ h) for h in self.perm_matrices],
                dtype=torch.long
            ).to(device)


def get_permutation_equivariance(equi_lang):
    """Helper function to construct and return an equivariance group for a language"""
    if equi_lang.num_equivariant_words == 0:
        return TrivialGroup(num_letters=equi_lang.n_words)

    return CircularShift(num_letters=equi_lang.n_words,
                         num_equivariant=equi_lang.num_equivariant_words,
                         first_equivariant=equi_lang.num_fixed_words + 1)
