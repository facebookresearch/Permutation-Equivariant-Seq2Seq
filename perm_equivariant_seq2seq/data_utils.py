from io import open
import unicodedata
import re
import os

from perm_equivariant_seq2seq.symmetry_groups import LanguageInvariance
from perm_equivariant_seq2seq.language_utils import Language, InvariantLanguage, EquivariantLanguage


SOS_token = 0
EOS_token = 1


"""
    Example: Enlgish-French data handling
"""


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def get_fre_eng(path='/Users/gordonjo/Downloads/fra-eng/fra.txt'):
    # Read the file and split into lines
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    input_lang = Language('english')
    output_lang = Language('french')
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    return input_lang, output_lang, pairs


"""
    SCAN Data handling
"""


def normalize_string_scan(s):
    # s += '.'
    s = re.sub(r"I_", r"", s)
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def read_scan_data(path):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string_scan(s) for s in l.split("IN: ")[-1].split(' OUT: ')] for l in lines]
    return pairs


def get_invariant_scan_languages(pairs, invariances):
    # Initialize language classes
    input_lang = Language('commands')
    syntax_lang = InvariantLanguage('syntax', invariances)
    output_lang = Language('actions')
    # Set-up languages
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        syntax_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    # Return languages (including invariant syntax)
    return input_lang, syntax_lang, output_lang


def get_equivariant_scan_languages(pairs, input_equivariances, output_equivariances):
    # Initialize language classes
    input_lang = EquivariantLanguage('commands', input_equivariances)
    output_lang = EquivariantLanguage('actions', output_equivariances)
    # Set up languages
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    # Manipulate indices
    input_lang.rearrange_indices()
    output_lang.rearrange_indices()
    return input_lang, output_lang


def get_scan_split(split=None):
    assert split in ['simple', 'add_jump', 'length_generalization', 'around_right', 'opposite_right'], \
        "Please choose valid experiment split"
    DATA_DIR = '../seq2seq-generalization/SCAN/'

    # Simple (non-generalization) split
    if split == 'simple':
        dir_path = os.path.join(DATA_DIR, 'simple_split')

        train_path = os.path.join(dir_path, 'tasks_train_simple.txt')
        test_path = os.path.join(dir_path, 'tasks_test_simple.txt')

    # Add jump generalization split
    elif split == 'add_jump':
        dir_path = os.path.join(DATA_DIR, 'add_prim_split')

        train_path = os.path.join(dir_path, 'tasks_train_addprim_jump.txt')
        test_path = os.path.join(dir_path, 'tasks_test_addprim_jump.txt')

    # Add "{around, opposite} right" template split
    elif split in ['around_right', 'opposite_right']:
        dir_path = os.path.join(DATA_DIR, 'template_split')

        train_path = os.path.join(dir_path, 'tasks_train_template_%s.txt' % split)
        test_path = os.path.join(dir_path, 'tasks_test_template_%s.txt' % split)

    # Add length generalization split
    elif split == 'length_generalization':
        dir_path = os.path.join(DATA_DIR, 'length_split')

        train_path = os.path.join(dir_path, 'tasks_train_length_.txt')
        test_path = os.path.join(dir_path, 'tasks_test_length.txt')

    # Load data
    training_pairs = read_scan_data(train_path)
    test_pairs = read_scan_data(test_path)
    return training_pairs, test_pairs


if __name__ == '__main__':
    VERB_INVARIANCE = LanguageInvariance(['jump', 'run', 'walk', 'look'], 'verb')
    DIRECTION_INVARIANCE = LanguageInvariance(['right', 'left'], 'direction')
    CONJUNCTION_INVARIANCE = LanguageInvariance(['and', 'after'], 'conjunction')
    ADVERB_INVARANCE = LanguageInvariance(['once', 'twice', 'thrice'], 'adverb')
    ALL_INVARIANCES = [VERB_INVARIANCE, DIRECTION_INVARIANCE, CONJUNCTION_INVARIANCE, ADVERB_INVARANCE]
    commands, command_syntax, actions, train_pairs, test_pairs = get_scan_split('simple', [VERB_INVARIANCE])
