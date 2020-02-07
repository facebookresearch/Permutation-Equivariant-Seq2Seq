# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import six
import abc

import torch
import torch.nn.functional as F

from perm_equivariant_seq2seq.models import AbsSeq2Seq
from perm_equivariant_seq2seq.g_rnn import EquiRNN, EquivariantDecoderRNN, EquivariantAttnDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1


class GroupAwareSeq2Seq(six.with_metaclass(abc.ABCMeta, AbsSeq2Seq)):
    """
    Abstract base class for equivariant sequence-2-sequence models

    Args:
        input_symmetry_group (PermutationSymmetry): Symmetry group for source language
        output_symmetry_group (PermutationSymmetry): Symmetry group for target language
        input_language (Language): Source language
        encoder_hidden_size (int): Encoder embedding dimensionality
        decoder_hidden_size (int): Decoder embedding dimensionality
        output_language (Language): Target language
        layer_type (str): Type of recurrent layer to be used
        use_attention (bool): Decoder uses attentive mechanisms
        drop_rate (0 <= float <= 1): Dropout rate to use in encoder / decoder
        bidirectional (bool): Encoder uses bidirectional mechanisms
        num_layers (int): number of hidden layers in recurrent modules
    """
    def __init__(self,
                 input_symmetry_group,
                 output_symmetry_group,
                 input_language,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 output_language,
                 layer_type,
                 use_attention=False,
                 drop_rate=0.,
                 bidirectional=False,
                 num_layers=1):
        super(GroupAwareSeq2Seq, self).__init__(input_language=input_language,
                                                encoder_hidden_size=encoder_hidden_size,
                                                decoder_hidden_size=decoder_hidden_size,
                                                output_language=output_language,
                                                layer_type=layer_type,
                                                use_attention=use_attention,
                                                drop_rate=drop_rate,
                                                bidirectional=bidirectional,
                                                num_layers=num_layers)
        self.input_symmetry_group = input_symmetry_group
        self.output_symmetry_group = output_symmetry_group

        self.encoder = EquiRNN(symmetry_group=self.input_symmetry_group,
                               input_size=self.input_language.n_words,
                               hidden_size=self.encoder_hidden_size,
                               cell_type=self.layer_type,
                               bidirectional=self.bidirectional,
                               nonlinearity='tanh')

    def forward(self,
                input_tensor,
                syntax_tensor=None,
                target_tensor=None,
                use_teacher_forcing=False):
        """Implements the forward pass through the model

        Args:
            input_tensor (torch.tensor): Input sequence in one-hot representation (T x |Vin|)
            syntax_tensor (torch.tensor): Optional Input sequence in syntax one-hot representation (Tin x |Vin|)
            target_tensor (torch.tensor): Optional Target sequence in one-hot representation (Tout x |Vout|)
            use_teacher_forcing (bool): Use teacher forcing when decoding (only during training, requires target)
        Returns:
            (torch.tensor) Sequence of softmax probability distributions over output vocabulary
        """
        # Initialize encoder hidden state
        encoder_hidden = self.encoder.init_hidden()

        # Convert sentences to sequence of one-hot vectors
        input_tensor = F.one_hot(input_tensor.t(), self.input_language.n_words).type(torch.float64)

        # Encode input sequence
        encoder_output, encoder_hidden = self.encoder(input_tensor, encoder_hidden)
        encoder_output = self._transfer_encoder_outputs(encoder_output)
        decoder_hidden = self._transfer_hidden(encoder_hidden)
        return self.decode(decoder_hidden, encoder_output, use_teacher_forcing, target_tensor)

    def decode(self,
               decoder_hidden,
               encoder_output,
               use_teacher_forcing=False,
               target_tensor=None):
        """Wrapper around usage of decoder to allow teacher forcing and code sharing for inheriting classes

        Args:
            decoder_hidden (torch.tensor): Initial hidden state for the decoder (last hidden state from encoder)
            encoder_output (torch.tensor): Hidden states from encoding of input sequence
            use_teacher_forcing (bool): Whether to use teacher forcing (only at train time)
            target_tensor (torch.tensor): If using teacher forcing, must have true target tensor
        Returns:
            (torch.tensor) Sequence of categorical distributions over output vocabulary
        """
        # Initialize some variables necessary for decoding
        _, decoder_input, decoder_outputs = self._init_forward_pass()

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            target_length = self.check_target_length(target_tensor)
            for di in range(target_length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                     decoder_hidden,
                                                                                     encoder_output)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                decoder_outputs[di] = decoder_output
                decoder_input = self._arrange_decoder_input_from_int(target_tensor[di])

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.max_length):
                if self.use_attention:
                    decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input,
                                                                                     decoder_hidden,
                                                                                     encoder_output)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                topv, topi = decoder_output.topk(1)
                decoder_outputs[di] = decoder_output
                decoder_input = topi.squeeze().detach()  # detach from history as input

                if decoder_input.item() == EOS_token:
                    break
                else:
                    decoder_input = self._arrange_decoder_input_from_int(decoder_input.unsqueeze(0))

        return decoder_outputs

    def _init_forward_pass(self):
        """Helper function for pre-computations required for forward pass"""
        encoder_hidden, decoder_input, decoder_output = super(GroupAwareSeq2Seq, self).init_forward_pass()
        decoder_input = F.one_hot(decoder_input[0], self.output_language.n_words).type(torch.float64)
        return encoder_hidden, decoder_input, decoder_output

    def _arrange_hidden(self, hidden):
        return hidden

    def _transfer_encoder_outputs(self, encoder_output):
        return encoder_output

    def _arrange_decoder_input_from_int(self, decoder_input):
        return F.one_hot(decoder_input, self.output_language.n_words).type(torch.float64)


class EquiSeq2Seq(GroupAwareSeq2Seq):
    """Implementation of "fully-supervised" Equivariant RNN. Fully supervised implies that symmetries are known (and the
    same) for the input and output languages. Both input / output languages must be of of type EquivariantLanguage, and
    symmetry_group must be of type CircularShift.

    Args:
        input_symmetry_group (PermutationSymmetry): Symmetry group for source language
        output_symmetry_group (PermutationSymmetry): Symmetry group for target language
        input_language (Language): Source language
        encoder_hidden_size (int): Encoder embedding dimensionality
        decoder_hidden_size (int): Decoder embedding dimensionality
        output_language (Language): Target language
        layer_type (str): Type of recurrent layer to be used
        use_attention (bool): Decoder uses attentive mechanisms
        drop_rate (0 <= float <= 1): Dropout rate to use in encoder / decoder
        bidirectional (bool): Encoder uses bidirectional mechanisms
        num_layers (int): number of hidden layers in recurrent modules
    """
    def __init__(self,
                 input_symmetry_group,
                 output_symmetry_group,
                 input_language,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 output_language,
                 layer_type,
                 use_attention=False,
                 drop_rate=0.,
                 bidirectional=False,
                 num_layers=1):

        super(EquiSeq2Seq, self).__init__(input_symmetry_group=input_symmetry_group,
                                          output_symmetry_group=output_symmetry_group,
                                          input_language=input_language,
                                          encoder_hidden_size=encoder_hidden_size,
                                          decoder_hidden_size=decoder_hidden_size,
                                          output_language=output_language,
                                          layer_type=layer_type,
                                          use_attention=use_attention,
                                          drop_rate=drop_rate,
                                          bidirectional=bidirectional,
                                          num_layers=num_layers)

        self.decoder_hidden_size = self.encoder_hidden_size * 2 if self.bidirectional else self.encoder_hidden_size
        decoder_model = EquivariantAttnDecoder if self.use_attention else EquivariantDecoderRNN

        self.decoder = decoder_model(hidden_size=self.decoder_hidden_size,
                                     output_size=self.output_language.n_words,
                                     symmetry_group=self.output_symmetry_group,
                                     cell_type=self.layer_type,
                                     nonlinearity='tanh')
