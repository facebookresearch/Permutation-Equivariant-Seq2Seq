# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from perm_equivariant_seq2seq.symmetry_groups import LanguageInvariance


class Language:
    """Object to keep track of languages to be translated.

    Args:
        name: (string) Name of language being used
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        """Process a sentence and add words to language vocabulary"""
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """Add a word to the language vocabulary"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class InvariantLanguage(Language):
    """Object to track a language with a fixed set of invariances

    Args:
        name (string): Name of language being used
        invariances (list::invariances): A list of invariance objects representing language invariances.
    """
    def __init__(self, name, invariances):
        super(InvariantLanguage, self).__init__(name)
        self.invariances = invariances

    def add_word(self, word):
        """Add a word to language vocabulary"""
        word = self.map_word(word)
        super(InvariantLanguage, self).add_word(word)

    def map_word(self, word):
        """Map a word to its equivalence class"""
        for invariance in self.invariances:
            word = invariance.map_word(word)
        return word

    def map_sentence(self, sentence):
        """Process a sentence and map all words to their equivalence classes"""
        return ' '.join([self.map_word(word) for word in sentence.split(' ')])


class EquivariantLanguage(Language):
    """Object to track a language with a fixed (and known) set of equivariances

    Args:
        name (string): Name of language being used
        equivariant_words (list::strings): List of words in language that are equivariant
    """
    def __init__(self, name, equivariant_words):
        super(EquivariantLanguage, self).__init__(name)
        self.equivariant_words = equivariant_words

    def rearrange_indices(self):
        """Rearrange the language indexing such that the first N words after the

        Returns:
            None
        """
        num_fixed_words = 2
        other_words = [w for w in self.word2index if w not in self.equivariant_words]
        for idx, word in enumerate(self.equivariant_words):
            w_idx = idx + num_fixed_words
            self.word2index[word] = w_idx
            self.index2word[w_idx] = word
        for idx, word in enumerate(other_words):
            w_idx = idx + num_fixed_words + self.num_equivariant_words
            self.word2index[word] = w_idx
            self.index2word[w_idx] = word

    @property
    def num_equivariant_words(self):
        return len(self.equivariant_words)

    @property
    def num_fixed_words(self):
        return 2

    @property
    def num_other_words(self):
        return len([w for w in self.word2index if w not in self.equivariant_words])


# Define SCAN language invariances
VERB_INVARIANCE = LanguageInvariance(['jump', 'run', 'walk', 'look'], 'verb')
DIRECTION_INVARIANCE = LanguageInvariance(['right', 'left'], 'direction')
CONJUNCTION_INVARIANCE = LanguageInvariance(['and', 'after'], 'conjunction')
ADVERB_INVARIANCE = LanguageInvariance(['once', 'twice', 'thrice'], 'adverb')
OTHER_INVARIANCE = LanguageInvariance(['around', 'opposite'], 'other')


def get_invariances(args):
    """Helper function to store some standard equivariances"""
    invariances = []
    if args.verb_invariance:
        invariances.append(VERB_INVARIANCE)
    if args.direction_invariance:
        invariances.append(DIRECTION_INVARIANCE)
    if args.conjunction_invariance:
        invariances.append(CONJUNCTION_INVARIANCE)
    if args.adverb_invariance:
        invariances.append(ADVERB_INVARIANCE)
    if args.other_invariance:
        invariances.append(OTHER_INVARIANCE)
    return invariances
