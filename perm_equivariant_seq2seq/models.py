# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import six
import abc

import torch
import torch.nn as nn

from perm_equivariant_seq2seq.modules import EncoderRNN, DecoderRNN, AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 50
SOS_token = 0
EOS_token = 1


class AbsSeq2Seq(six.with_metaclass(abc.ABCMeta, nn.Module)):
    """Abstract Seq2Seq base model

    Args:
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
                 input_language,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 output_language,
                 layer_type,
                 use_attention,
                 drop_rate,
                 bidirectional,
                 num_layers):
        super(AbsSeq2Seq, self).__init__()

        self.max_length = MAX_LENGTH
        self.input_language = input_language
        self.input_size = self.input_language.n_words
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_language = output_language
        if layer_type not in ['GRU', 'LSTM', 'RNN', 'GRNN', 'GGRU', 'GLSTM']:
            raise NotImplementedError("Supported cells: '(G)RNN', '(G)GRU', or '(G)LSTM'")
        self.layer_type = layer_type
        self.output_size = self.output_language.n_words
        self.use_attention = use_attention
        self.drop_rate = drop_rate
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

    @abc.abstractmethod
    def forward(self,
                input_tensor,
                syntax_tensor=None,
                target_tensor=None,
                use_teacher_forcing=False):
        """Forward pass for model. Conditions on sentence in input language and generates a sentence in output language

        Args:
            input_tensor (torch.tensor): Tensor representation (1-hot) of sentence in input language
            syntax_tensor (torch.tensor, optional): Tensor representation (1-hot) of sentence in input language syntax
            target_tensor (torch.tensor, optional): Tensor representation (1-hot) of target sentence in output language
            use_teacher_forcing (bool, optional): Indicates if true word is used as input to decoder. Default: False
        Returns:
            (torch.tensor) Tensor representation (softmax probabilities) of target sentence in output language
        """

    def init_forward_pass(self):
        """Helper function initialize required objects on forward pass"""
        encoder_hidden = self.encoder.init_hidden()
        decoder_outputs = torch.zeros(MAX_LENGTH, self.decoder.output_size, device=device)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        return encoder_hidden, decoder_input, decoder_outputs

    def _arrange_hidden(self, hidden):
        """Reshape and rearrange the final encoder state for initialization of decoder state

        Args:
            hidden: (torch.tensor) Final output of the encoder
        Returns:
            (torch.tensor) Final output arranged for decoder
        """
        hidden = hidden.view(self.num_layers, self.num_directions, 1, -1)
        return hidden[-1].view(1, 1, -1)

    def _transfer_hidden(self, hidden):
        """Rearrange the final hidden state from encoder to be passed to decoder

        Args:
            hidden: (torch.tensor) final hidden state of encoder module
        Returns:
            (torch.tensor) initial hidden state to be passed to decoder
        """
        if 'GRU' in self.layer_type or 'RNN' in self.layer_type:
            return self._arrange_hidden(hidden)
        elif 'LSTM' in self.layer_type:
            return tuple([self._arrange_hidden(h) for h in hidden])
        else:
            raise NotImplementedError

    def check_target_length(self, target_tensor):
        """Helper function to determine length of input sequence"""
        if target_tensor is not None:
            target_length = target_tensor.size(0)
            assert target_length <= MAX_LENGTH, print("Max length exceeded. Max Length: %s, Target length: %s"
                                                      % (MAX_LENGTH, target_length))
            return target_length
        else:
            return None

    @property
    def num_params(self):
        """Count number of parameters in model"""
        count = 0
        for param in self.parameters():
            count += torch.tensor(param.shape).prod()
        return count


class BasicSeq2Seq(AbsSeq2Seq):
    """Standard implementation of sequence to sequence model for translation (potentially using attention)

    Args:
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
                 input_language,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 output_language,
                 layer_type,
                 use_attention,
                 drop_rate,
                 bidirectional,
                 num_layers):
        super(BasicSeq2Seq, self).__init__(input_language=input_language,
                                           encoder_hidden_size=encoder_hidden_size,
                                           decoder_hidden_size=decoder_hidden_size,
                                           output_language=output_language,
                                           layer_type=layer_type,
                                           use_attention=use_attention,
                                           drop_rate=drop_rate,
                                           bidirectional=bidirectional,
                                           num_layers=num_layers)

        self.encoder = EncoderRNN(input_size=self.input_size,
                                  hidden_size=self.encoder_hidden_size,
                                  layer_type=self.layer_type,
                                  bidirectional=self.bidirectional,
                                  num_layers=self.num_layers)
        if self.use_attention:
            self.decoder = AttnDecoderRNN(hidden_size=self.decoder_hidden_size * self.num_directions,
                                          output_size=self.output_size,
                                          layer_type=self.layer_type,
                                          dropout_p=self.drop_rate)
        else:
            self.decoder = DecoderRNN(hidden_size=self.decoder_hidden_size,
                                      output_size=self.output_size,
                                      layer_type=self.layer_type)

    def forward(self,
                input_tensor,
                syntax_tensor=None,
                target_tensor=None,
                use_teacher_forcing=False):
        """Forward pass for model. Conditions on sentence in input language and generates a sentence in output language

        Args:
            input_tensor (torch.tensor): Tensor representation (1-hot) of sentence in input language
            syntax_tensor (torch.tensor, optional): Tensor representation (1-hot) of sentence in input language syntax
            target_tensor (torch.tensor, optional): Tensor representation (1-hot) of target sentence in output language
            use_teacher_forcing (bool, optional): Indicates if true word is used as input to decoder. Default: False
        Returns:
            (torch.tensor) Tensor representation (softmax probabilities) of target sentence in output language
        """
        # Some bookkeeping and preparaion for forward pass
        encoder_hidden, decoder_input, decoder_outputs = self.init_forward_pass()

        encoder_output, encoder_hidden, _ = self.encoder(input_tensor, encoder_hidden)
        decoder_hidden = self._transfer_hidden(encoder_hidden)

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
                decoder_input = target_tensor[di]  # Teacher forcing

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

        return decoder_outputs
