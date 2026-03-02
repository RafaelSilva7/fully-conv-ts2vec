import math
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _reverse_repeat_tuple
from typing import Union, Literal, Callable


def linear_interpolation(x: torch.Tensor,
                         offsets: torch.Tensor,
                         kernel_size: int,
                         dilation: int,
                         stride: int,
                         dilated_positions = None,
                         device: str = 'cpu',
                         unconstrained: bool = False,
                         _test: bool = False) -> None:
    """ Linear interpolation base method used as a form to calculate each
    new position for the kernel grid in deformable convolutions.
    """
    assert x.device == offsets.device, 'x and offsets must be on the same device.'
    kernel_rfield = dilation * (kernel_size - 1) + 1 # Calculate the receptive field
    # Every index in x we need to consider
    if dilated_positions == None:
        dilated_positions = torch.linspace(0, kernel_rfield - 1, kernel_size, device=offsets.device, dtype=offsets.dtype)

    max_t0 = (offsets.shape[-2] - 1) * stride
    t0s = torch.linspace(0, max_t0, offsets.shape[-2], device=offsets.device, dtype=offsets.dtype).unsqueeze(-1)
    dilated_offsets_repeated = dilated_positions + offsets

    T = t0s + dilated_offsets_repeated
    if not unconstrained:
        T = torch.max(T, t0s)
        T = torch.min(T, t0s + torch.max(dilated_positions))
    else:
        T = torch.clamp(T, 0.0, float(x.shape[-1]))

    with torch.no_grad():
        U = torch.floor(T).to(torch.long)
        U = torch.clamp(U, min=0, max=x.shape[2] - 2)

        if _test:
            print('U: ', U.shape)
        
        U = torch.stack([U, U + 1], dim=-1)
        if U.shape[1] < x.shape[1]:
            U = U.repeat(1, x.shape[1], 1, 1, 1)

        if _test:
            print('U: ', U.shape)

    x = x.unsqueeze(-1).repeat(1, 1, 1, U.shape[-1])
    x = torch.stack([x.gather(index=U[:,:,:,i,:], dim=2) for i in range(U.shape[-2])], dim=-1)

    G = torch.max(torch.zeros(U.shape, device=device),
                  1 - torch.abs(U - T.unsqueeze(-1))) # Batch x Groups x Out Length x Kernel RField x Kernel Size
    
    if _test:
        print('G: ', G.shape)

    mx = torch.multiply(G, x.moveaxis(-2, -1))
    return torch.sum(mx, axis=-1) # Batch x Channels x Out Length x Kernel Size


class LeakySineLU(nn.Module):
    """
    This class implements the activation proposed on [1].

    ### Use:
    >>> x = torch.randn((120, 12))
    >>> activation = LeakySineLU()
    >>> y = activation(x)

    ### References:
    [1] de Medeiros Júnior, J.G.B., de Mitri, A.G., Silva, D.F. (2025).
        Semi-periodic Activation for Time Series Classification.
        In: Paes, A., Verri, F.A.N. (eds) Intelligent Systems. BRACIS 2024.
        Lecture Notes in Computer Science(), vol 15415.
        Springer, Cham. https://doi.org/10.1007/978-3-031-79038-6_6.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x > 0, torch.sin(x) ** 2 + x, (0.5 * torch.sin(x) ** 2) + x
        )


class GlobalLayerNormalization(nn.Module):

    def __init__(self, channel_size: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, 1, channel_size))
        self.beta = nn.Parameter(torch.Tensor(1, 1, channel_size))

        self.EPS = 1e-9

        self.reset_parameters()


    def reset_parameters(self) -> None:
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = (torch.pow(x -mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)

        gln_x = self.gamma * (x - mean) / torch.pow(var + self.EPS, 0.5) + self.beta
        return gln_x


class DeformableConvolution1d(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[int, Literal['valid', 'same']] = 'valid',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'reflect',
                 device: str = 'cpu',
                 interpolation_method: Callable = linear_interpolation,
                 unconstrained: str = None,
                 *args,
                 **kwargs) -> None:
        self.device = device
        self.interpolation_method = interpolation_method
        padding_ = padding if isinstance(padding, str) else _single(padding)
        stride_ = _single(stride)
        dilation_ = _single(dilation)
        kernel_size_ = _single(kernel_size)

        super().__init__(*args, **kwargs)

        if groups < 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('input channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out channels must be divisible by groups')
        
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError('invalid padding string, you must use valid or same')
            if padding == 'same' and any(s != 1 for s in stride_):
                raise ValueError('padding=same is not supported for strided convolutions')
            
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError('invalid padding mode, you must use zeros, reflect, replicate or circular')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding_
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding == 'same':
                for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.weigth = Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        self.dilated_positions = torch.linspace(0, dilation * kernel_size - dilation, kernel_size)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        if not unconstrained == None:
            self.unconstrained = unconstrained
        
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weigth, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weigth)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        s += ', dilation={dilation}'

        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'

        return s.format(**self.__dict__)
    
    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def forward(self, x: torch.Tensor, offsets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        in_shape = x.shape

        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        elif self.padding == 'same':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode='constant', value=0)

        if not self.device == offsets.device:
            self.device = offsets.device
        
        if self.dilated_positions.device != self.device:
            self.dilated_positions = self.dilated_positions.to(self.device)

        if 'unconstrained' in self.__dict__.keys():
            x = self.interpolation_method(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device,
                unconstrained=self.unconstrained
            )
        else:
            x = self.interpolation_method(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                offsets=offsets,
                stride=self.stride,
                dilated_positions=self.dilated_positions,
                device=self.device
            )

        mask = mask.contiguous().permute((0, 2, 1)).unsqueeze(dim=1)
        mask = torch.cat([mask for _ in range(x.size(1))], dim=1)
        
        x *= mask
        x = x.flatten(-2, -1)
        output = F.conv1d(x, weight=self.weigth, bias=self.bias, stride=self.kernel_size, groups=self.groups)

        if self.padding == 'same':
            assert in_shape[-1] == output.shape[-1], f'input length {in_shape} and output length {output.shape} do not match'

        return output
    

class PackedDeformableConvolution1d(DeformableConvolution1d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[int, Literal['valid', 'same']] = 'valid',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'reflect',
                 offset_groups: int = 1,
                 device: str = 'cpu',
                 interpolation_method: Callable = linear_interpolation,
                 unconstrained: str = None,
                 *args, **kwargs) -> None:
        
        assert offset_groups in [1, in_channels], 'offset groups only implemented for 1 or in_channels'

        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias,
                         padding_mode,
                         device,
                         interpolation_method,
                         unconstrained,
                         *args, **kwargs)
        
        self.offset_groups = offset_groups
        self.offset_dconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            padding=padding,
            padding_mode=padding_mode,
            bias=False
        )
        self.offset_dconv_norm = GlobalLayerNormalization(in_channels)
        self.offset_dconv_prelu = nn.PReLU()

        self.offset_pconv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=kernel_size * offset_groups,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.offset_pconv_norm = GlobalLayerNormalization(kernel_size * offset_groups)
        self.offset_pconv_prelu = nn.PReLU()

        self.modulation_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=kernel_size,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=False
        )
        self.modulation_conv_softmax = nn.Softmax(dim=1)

        self.device = device
        self.to(device)
        
        torch.nn.init.constant_(self.offset_dconv.weight, 0.)
        torch.nn.init.constant_(self.offset_pconv.weight, 0.)

        # if bias:
        #     torch.nn.init.constant_(self.offset_dconv.bias, 1.)
        #     torch.nn.init.constant_(self.offset_pconv.bias, 1.)
            
        self.offset_dconv.register_backward_hook(self._set_lr)
        self.offset_pconv.register_backward_hook(self._set_lr)

        torch.nn.init.constant_(self.modulation_conv.weight, 0.5)
        self.modulation_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x: torch.Tensor, with_offsets: bool = False) -> torch.Tensor:
        
        offsets = self.offset_dconv(x)
        offsets = self.offset_dconv_norm(self.offset_dconv_prelu(offsets).moveaxis(1, 2)).moveaxis(2, 1)
        
        m = self.modulation_conv_softmax(self.modulation_conv(x))
        
        self.device = x.device
        
        assert str(x.device) == str(self.device), 'x and the deformable conv must be on same device'
        # assert str(x.device) == str(offsets.device), 'x and offsets must be on same device'
        
        offsets = self.offset_pconv(x)
        offsets = self.offset_pconv_norm(
            self.offset_pconv_prelu(offsets).moveaxis(1, 2)
        ).moveaxis(2, 1)
        offsets = offsets.unsqueeze(0).chunk(self.offset_groups, dim=2)
        offsets = torch.vstack(offsets).moveaxis((0, 2), (1, 3))
        
        if with_offsets:
            return super().forward(x, offsets, mask=m), offsets
        else:
            return super().forward(x, offsets, mask=m)
        

class WeightedConv1D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: str | int = 'same',
                 bias: bool = False,
                 weight_lr_ratio: float = 0.1) -> None:
        """
        Args:
            weight_lr_ratio (float): learning rate ratio for each input channel related to the default network
                                     lr. Default is 0.1 (10%).
        """
        super().__init__()

        self.channel_weights = nn.Parameter(torch.ones(1, in_channels, 1))

        self.channel_weights.register_hook(
            lambda grad: grad * weight_lr_ratio
        )

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.channel_weights)


class SEBlock1d(nn.Module):
    def __init__(self, channel: int, reduction: int =16):
        """
        Args:
            channel (int): Number of channels
            reduction (int): Reduction factor for bottleneck (default 16).
        """
        super().__init__()

        # Transform (B, C L) -> (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Excitation Mechanism
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid() # Saída entre 0 e 1 (a importância de cada canal)
        )

    def forward(self, x):
        b, c, _ = x.size()
        
        # --- Squeeze ---
        y = self.avg_pool(x)
        y = y.view(b, c)
        
        # --- Excitation ---
        # Shape: (b, c)
        y = self.fc(y)
        # Shape: (b, c, 1)
        y = y.view(b, c, 1)
        
        # --- Scale ---
        return x * y


class SEConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, reduction=16):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        # TODO: maybe modify the activation here
        self.relu = nn.ReLU()
        
        self.se = SEBlock1d(channel=out_channels, reduction=reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        
        x = self.se(x)
        
        return x
    

class DisjoinEncoder(nn.Module):
    def __init__(self, channel_size, emb_size, kernel_size, rep_size = None):
        super().__init__()
        self.temporal_CNN = nn.Sequential(nn.Conv2d(1, emb_size, kernel_size=[1, kernel_size], padding='same'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        self.spatial_CNN = nn.Sequential(nn.Conv2d(emb_size, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                         nn.BatchNorm2d(emb_size),
                                         nn.GELU())

        # self.rep_CNN = nn.Sequential(nn.Conv1d(emb_size, rep_size, kernel_size=3),
        #                              nn.BatchNorm1d(rep_size),
        #                              nn.GELU())
        # Initialize the weights
        self.initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x)

        # x = self.rep_CNN()
        return x.squeeze(dim=2)

    def initialize_weights(self):
        # Custom weight initialization, you can choose different methods
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layer weights using Xavier/Glorot initialization
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    # Initialize biases with zeros
                    init.constant_(m.bias, 0)