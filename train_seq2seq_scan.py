# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
import argparse
import os

import torch
import torch.nn as nn

import perm_equivariant_seq2seq.utils as utils
from perm_equivariant_seq2seq.models import BasicSeq2Seq
from perm_equivariant_seq2seq.data_utils import get_scan_split
from perm_equivariant_seq2seq.utils import tensors_from_pair
from perm_equivariant_seq2seq.data_utils import get_invariant_scan_languages

"""
[1]: Lake and Baroni 2019: Generalization without systematicity: On the 
compositional skills of seq2seq networks
[2]: Bahdanau et al. 2014: Neural machine translation by jointly learning to 
align and translate
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1

# Parse command-line arguments
parser = argparse.ArgumentParser()
# Model options
parser.add_argument('--layer_type', 
                    choices=['LSTM', 'GRU', 'RNN'], 
                    default='LSTM',
                    help='Type of rnn layers to be used for recurrent components')
parser.add_argument('--hidden_size', 
                    type=int, 
                    default=64, 
                    help='Number of hidden units in encoder / decoder')
parser.add_argument('--semantic_size', 
                    type=int, 
                    default=64, 
                    help='Dimensionality of semantic embedding')
parser.add_argument('--num_layers', 
                    type=int, 
                    default=1, 
                    help='Number of hidden layers in encoder')
parser.add_argument('--use_attention', 
                    dest='use_attention', 
                    default=False, 
                    action='store_true',
                    help="Boolean to use attention in the decoder")
parser.add_argument('--bidirectional', 
                    dest='bidirectional', 
                    default=False, 
                    action='store_true',
                    help="Boolean to use bidirectional encoder")
parser.add_argument('--drop_rate', 
                    type=float, 
                    default=0.1, 
                    help="Dropout drop rate (not keep rate)")
# Optimization and training hyper-parameters
parser.add_argument('--split', 
                    choices=[None, 'simple', 'add_jump', 'length_generalization'],
                    help='Each possible split defines a different experiment as proposed by [1]')
parser.add_argument('--validation_size', 
                    type=float, 
                    default=0.2,
                    help='Validation proportion to use for early-stopping')
parser.add_argument('--n_iters', 
                    type=int, 
                    default=200000, 
                    help='number of training iterations')
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=1e-4,
                    help='init learning rate')
parser.add_argument('--teacher_forcing_ratio', 
                    type=float, 
                    default=0.5)
parser.add_argument('--save_dir', 
                    type=str, 
                    default='./models/', 
                    help='Top-level directory for saving experiment')
parser.add_argument('--print_freq', 
                    type=int, 
                    default=1000, 
                    help='Frequency with which to print training loss')
parser.add_argument('--save_freq', 
                    type=int, 
                    default=20000, 
                    help='Frequency with which to save models during training')
args = parser.parse_args()

args.save_path = os.path.join(args.save_dir,
                              '%s' % args.split,
                              '%s' % args.layer_type,
                              'rnn_%s_hidden_%s_semantic_%s_layers_%s' % \
                              (args.layer_type,
                               args.hidden_size,
                               args.semantic_size,
                               args.num_layers))
# Create model directory
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

def train(input_tensor,
          target_tensor,
          model_to_train,
          enc_optimizer,
          dec_optimizer,
          loss_fn,
          teacher_forcing_ratio):
    """Perform one training iteration for the model

    Args:
        input_tensor: (torch.tensor) Tensor representation (1-hot) of sentence 
        in input language
        target_tensor: (torch.tensor) Tensor representation (1-hot) of target 
        sentence in output language
        model_to_train: (nn.Module: Seq2SeqModel) seq2seq model being trained
        enc_optimizer: (torch.optimizer) Optimizer object for model encoder
        dec_optimizer: (torch.optimizer) Optimizer object for model decoder
        loss_fn: (torch.nn.Loss) Loss object used for training
        teacher_forcing_ratio: (float) Ratio with which true word is used as 
        input to decoder
    Returns:
        (torch.scalar) Value of loss achieved by model at current iteration
    """
    model.train()
    # Forget gradients via optimizers
    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    model_output = model_to_train(input_tensor=input_tensor,
                                  target_tensor=target_tensor,
                                  use_teacher_forcing=use_teacher_forcing)
    train_loss = 0

    target_length = target_tensor.size(0)
    for di in range(target_length):
        decoder_output = model_output[di]
        train_loss += loss_fn(decoder_output[None, :], target_tensor[di])
        _, decoder_output_symbol = decoder_output.topk(1)
        if decoder_output_symbol.item() == EOS_token:
            break
    train_loss.backward()

    # Clip gradients by norm (5.) and take optimization step
    torch.nn.utils.clip_grad_norm_(model_to_train.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(model_to_train.decoder.parameters(), 5.)
    enc_optimizer.step()
    dec_optimizer.step()

    return train_loss.item() / target_length


def test_accuracy(model_to_test, pairs):
    """Test a model (metric: accuracy) on all pairs in _pairs_

    Args:
        model_to_test: (seq2seq) Model object to be tested
        pairs: (list::pairs) List of list of input/output language pairs
    Returns:
        (float) accuracy on test pairs
    """

    def sentence_correct(target, model_sentence):
        # First, extract sentence up to EOS
        _, sentence_ints = model_sentence.data.topk(1)
        # If there is no EOS token, take the complete list
        try:
            eos_location = (sentence_ints == EOS_token).nonzero()[0][0]
        except:
            eos_location = len(sentence_ints) - 2
        model_sentence = sentence_ints[:eos_location + 1]
        # Check length is correct
        if len(model_sentence) != len(target):
            return torch.tensor(0, device=device)
        else:
            correct = model_sentence == target
            return torch.prod(correct).to(device)

    accuracies = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            input_tensor, output_tensor = pair
            model_output = model_to_test(input_tensor=input_tensor)
            accuracies.append(sentence_correct(output_tensor, model_output))
    return torch.stack(accuracies).type(torch.float).mean()


if __name__ == '__main__':
    # Load data
    train_pairs, test_pairs = get_scan_split(split=args.split)
    commands, actions = get_invariant_scan_languages(train_pairs, invariances=[])

    # Initialize model
    model = BasicSeq2Seq(input_language=commands,
                         encoder_hidden_size=args.hidden_size,
                         decoder_hidden_size=args.semantic_size,
                         output_language=actions,
                         layer_type=args.layer_type,
                         use_attention=args.use_attention,
                         drop_rate=args.drop_rate,
                         bidirectional=args.bidirectional,
                         num_layers=args.num_layers)
    model.to(device)

    # Initialize optimizers
    encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), 
                                         lr=args.learning_rate)
    decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), 
                                         lr=args.learning_rate)

    # Split off validation set
    val_size = int(len(train_pairs) * args.validation_size)
    random.shuffle(train_pairs)
    train_pairs, val_pairs = train_pairs[val_size:], train_pairs[:val_size]

    # Convert data to torch tensors
    training_pairs = [tensors_from_pair(random.choice(train_pairs), commands, actions)
                      for i in range(args.n_iters)]
    training_eval = [tensors_from_pair(pair, commands, actions) 
                     for pair in train_pairs]
    validation_pairs = [tensors_from_pair(pair, commands, actions) 
                        for pair in val_pairs]
    testing_pairs = [tensors_from_pair(pair, commands, actions) 
                     for pair in test_pairs]

    # Initialize criterion
    criterion = nn.NLLLoss().to(device)

    # Initialize printing / plotting variables
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    # Enter training loop
    best_acc = 0.
    model_path = utils.create_exp_dir(args)
    for iteration in range(1, args.n_iters + 1):
        # Grab iteration translation triplet (input tensor, syntax tensor, output tensor)
        training_pair = training_pairs[iteration - 1]
        iteration_input, iteration_output = training_pair

        # Compute loss (and take one gradient step)
        loss = train(input_tensor=iteration_input,
                     target_tensor=iteration_output,
                     model_to_train=model,
                     enc_optimizer=encoder_optimizer,
                     dec_optimizer=decoder_optimizer,
                     loss_fn=criterion,
                     teacher_forcing_ratio=args.teacher_forcing_ratio)

        print_loss_total += loss
        plot_loss_total += loss

        # Print, plot, etc'
        if iteration % args.print_freq == 0:
            print_loss_avg = print_loss_total / args.print_freq
            print_loss_total = 0
            print('%s iterations: %s' % (iteration, print_loss_avg))

        if iteration % args.save_freq == 0:
            # save model if is better
            if args.validation_size > 0.:
                val_acc = test_accuracy(model, validation_pairs).item()
                if val_acc > best_acc:
                    best_acc = val_acc
                    save_path = os.path.join(model_path, 'best_validation.pt')
                    print('Best validation accuracy at iteration %s: %s' % (iteration + 1, val_acc))

    # Save fully trained model
    save_path = os.path.join(model_path, 'model_fully_trained.pt')
    torch.save(model.state_dict(), save_path)
