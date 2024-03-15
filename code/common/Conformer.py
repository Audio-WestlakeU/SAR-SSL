""" 
    Function:   Conformer 
    Refs:   Conformer: Convolution-augmented Transformer for Speech Recognition, INTERSPEECH 2020
            https://github.com/sooftware/conformer
"""
import numpy as np
import torch
import torch.nn as nn

from common.conformer.feed_forward import FeedForwardModule
from common.conformer.attention import MultiHeadedSelfAttentionModule
from common.conformer.convolution import ConformerConvModule, Conv2dSubampling
from common.conformer.modules import ResidualConnectionModule, Linear


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
            ),),
            ResidualConnectionModule(module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            ),),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sequential(inputs)

class ConformerEncoder(nn.Module):
    """
    Conformer encoder first processes the input with a convolution subsampling layer and then
    with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_layers: int = 6,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()
        # self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
        # self.input_projection = nn.Sequential(
        # Linear0(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
        # nn.Dropout(p=input_dropout_p),
        # )
        self.layers = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            ) for _ in range(num_layers)
        ])
 
    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: torch.Tensor, add_same_one) -> torch.Tensor:
        """
        Forward propagate a `inputs` for  encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        # outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        # outputs = self.input_projection(outputs)
        outputs = inputs + 0.0

        for layer in self.layers:
            # outputs = layer(inputs)
            outputs = layer(outputs)
            # temp = torch.mean(outputs[:,1:,:], dim=1)
            # outputs = outputs + temp[:,np.newaxis,:].expand(-1, outputs.shape[1], -1) # !!!
            if add_same_one:
                outputs_one = torch.mean(outputs, dim=1)
                outputs_one = outputs_one[:, np.newaxis, :].expand(outputs.shape[0], outputs.shape[1], outputs.shape[-1])
                outputs = outputs + outputs_one

        return outputs


# class Conformer(nn.Module):
#     """
#     Conformer: Convolution-augmented Transformer for Speech Recognition
#     The paper used a one-lstm Transducer decoder, currently still only implemented
#     the conformer encoder shown in the paper.

#     Args:
#         num_classes (int): Number of classification classes
#         input_dim (int, optional): Dimension of input vector
#         encoder_dim (int, optional): Dimension of conformer encoder
#         num_encoder_layers (int, optional): Number of conformer blocks
#         num_attention_heads (int, optional): Number of attention heads
#         feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
#         conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
#         feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
#         attention_dropout_p (float, optional): Probability of attention module dropout
#         conv_dropout_p (float, optional): Probability of conformer convolution module dropout
#         conv_kernel_size (int or tuple, optional): Size of the convolving kernel
#         half_step_residual (bool): Flag indication whether to use half step residual or not

#     Inputs: inputs, input_lengths
#         - **inputs** (batch, time, dim): Tensor containing input vector
#         - **input_lengths** (batch): list of sequence input lengths

#     Returns: outputs, output_lengths
#         - **outputs** (batch, out_channels, time): Tensor produces by conformer.
#         - **output_lengths** (batch): list of sequence output lengths
#     """
#     def __init__(
#             self,
#             num_classes: int,
#             input_dim: int = 80,
#             encoder_dim: int = 512,
#             num_encoder_layers: int = 17,
#             num_attention_heads: int = 8,
#             feed_forward_expansion_factor: int = 4,
#             conv_expansion_factor: int = 2,
#             input_dropout_p: float = 0.1,
#             feed_forward_dropout_p: float = 0.1,
#             attention_dropout_p: float = 0.1,
#             conv_dropout_p: float = 0.1,
#             conv_kernel_size: int = 31,
#             half_step_residual: bool = True,
#     ) -> None:
#         super(Conformer, self).__init__()
#         self.encoder = ConformerEncoder(
#             input_dim=input_dim,
#             encoder_dim=encoder_dim,
#             num_layers=num_encoder_layers,
#             num_attention_heads=num_attention_heads,
#             feed_forward_expansion_factor=feed_forward_expansion_factor,
#             conv_expansion_factor=conv_expansion_factor,
#             input_dropout_p=input_dropout_p,
#             feed_forward_dropout_p=feed_forward_dropout_p,
#             attention_dropout_p=attention_dropout_p,
#             conv_dropout_p=conv_dropout_p,
#             conv_kernel_size=conv_kernel_size,
#             half_step_residual=half_step_residual,
#         )
#         self.fc = Linear0(encoder_dim, num_classes, bias=False)

#     def count_parameters(self) -> int:
#         """ Count parameters of encoder """
#         return self.encoder.count_parameters()

#     def update_dropout(self, dropout_p) -> None:
#         """ Update dropout probability of model """
#         self.encoder.update_dropout(dropout_p)

#     def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward propagate a `inputs` and `targets` pair for training.

#         Args:
#             inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
#                 `FloatTensor` of size ``(batch, seq_length, dimension)``.
#             input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

#         Returns:
#             * predictions (torch.FloatTensor): Result of model predictions.
#         """
#         encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
#         outputs = self.fc(encoder_outputs)
#         outputs = nn.functional.log_softmax(outputs, dim=-1)
#         return outputs, encoder_output_lengths