# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1


def indexes_from_sentence(lang, sentence):
    """Extract all indices from a sentence

    Args:
        lang (Language): Language object from which sentence is derived
        sentence (list::string): List of words in sentence
    Returns:
        (list::int) List of integers representing sentence
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    """Convert a sentence to a torch.tensor

    Args:
        lang (Language): Language object from which sentence is derived
        sentence: (list::int) List of integers representing sentence
    Returns:
        (torch.tensor) Tensor representation of sentence
    """
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair, lang1, lang2):
    """Convert a translation pair to torch.tensor types

    Args:
        pair (list::list): List (pair) of sentences in string format
        lang1 (Language): Language object from which input sentence is derived
        lang2 (Language): Language object from which target sentence is derived
    Returns:
        (tuple::torch.tensors) Tensor representation of translation triplet (input, syntax, output)
    """
    input_tensor = tensor_from_sentence(lang1, pair[0])
    target_tensor = tensor_from_sentence(lang2, pair[1])
    return input_tensor, target_tensor


""" Saving and loading models """


def create_exp_dir(args):
    """Create a new experiment directory if one does not exist, save argparse object to directory

    Args:
        args (Argparse): Argparse object that defines the experiment
    Returns:
        None
    """
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    exists = True
    model_num = 0
    while exists:
        model_path = os.path.join(args.save_path, 'model%s' % model_num)
        if os.path.exists(model_path):
            model_num += 1
        else:
            exists = False
    os.makedirs(model_path)
    print('Experiment dir : {}'.format(model_path))
    args_path = os.path.join(model_path, 'commandline_args.txt')
    with open(args_path, 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    return model_path


def save_model(model, model_path):
    """Save a model to the experiment directory defined in args. If model exists in directory, increment model number

    Args
        model (Object.Model): Model object to be saved
        model_path (str): Directory to save model in
    Returns:
        None
    """
    torch.save(model.state_dict(), model_path)


def load_args_from_txt(parser, args_dir):
    """Load experiment command line arguments from text file

    Args:
        parser (Argparse): Argparse parser object to load commands into
        args_dir (string): Directory path of experiment args text file
    Returns:
        (dict) Dictionary containing command line arguments from experiment
    """
    with open(args_dir) as f:
        commands = f.read()
    list_of_lists = [command.split('=') for command in commands.split('\n')]
    flat_list_of_commands = [item for sublist in list_of_lists for item in sublist]
    return parser.parse_args(flat_list_of_commands)

