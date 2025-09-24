# # This file contains the ResNet architecture for the Federated Learning.
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def init_weights(net, init_type = 'xavier', init_param = 1.0):

#     def init_func(m):
#         classname = m.__class__.__name__
#         if classname.startswith('Conv') or classname == 'Linear':
#             if getattr(m, 'bias', None) is not None:
#                 init.constant_(m.bias, 0.0)
#             if getattr(m, 'weight', None) is not None:
#                 if init_type == 'normal':
#                     init.normal_(m.weight, 0.0, init_param)
#                 elif init_type == 'xavier':
#                     init.xavier_normal_(m.weight, gain=init_param)
#                 elif init_type == 'xavier_unif':
#                     init.xavier_uniform_(m.weight, gain=init_param)
#                 elif init_type == 'kaiming':
#                     init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
#                 elif init_type == 'kaiming_out':
#                     init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
#                 elif init_type == 'orthogonal':
#                     init.orthogonal_(m.weight, gain=init_param)
#                 elif init_type == 'default':
#                     if hasattr(m, 'reset_parameters'):
#                         m.reset_parameters()
#                 else:
#                     raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#         elif 'Norm' in classname:
#             if getattr(m, 'weight', None) is not None:
#                 m.weight.data.fill_(1)
#             if getattr(m, 'bias', None) is not None:
#                 m.bias.data.zero_()

#     net.apply(init_func)
#     return net

# # 先写PatchModules元类和ReparamModule，和你给的代码差不多，稍作简化：
# class PatchModules(type):
#     def __call__(cls, *args, **kwargs):
#         net = type.__call__(cls,*args, **kwargs)

#         w_modules_names = []
#         for m in net.modules():
#             for n, p in m.named_parameters(recurse=False):
#                 if p is not None:
#                     w_modules_names.append((m, n))

#         net._weights_module_names = tuple(w_modules_names)
#         net = net.to(device)

#         ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)
#         net._weights_numels = tuple(w.numel() for w in ws)
#         net._weights_shapes = tuple(w.shape for w in ws)

#         with torch.no_grad():
#             flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

#         for m, n in net._weights_module_names:
#             delattr(m, n)
#             m.register_buffer(n, None)

#         net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

#         return net


# class ReparamModule(nn.Module, metaclass=PatchModules):
#     def _apply(self, *args, **kwargs):
#         rv = super(ReparamModule, self)._apply(*args, **kwargs)
#         return rv

#     def get_param(self, clone=False):
#         if clone:
#             return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
#         return self.flat_w

#     @contextmanager
#     def unflatten_weight(self, flat_w):
#         ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
#         for (m, n), w in zip(self._weights_module_names, ws):
#             setattr(m, n, w)
#         yield
#         for m, n in self._weights_module_names:
#             setattr(m, n, None)

#     def forward_with_param(self, inp, new_w):
#         with self.unflatten_weight(new_w):
#             return nn.Module.__call__(self, inp)

#     def __call__(self, inp):
#         return self.forward_with_param(inp, self.flat_w)

#     @contextmanager
#     def reset(self,inplace=True):
#         if inplace:
#             flat_w = self.flat_w
#         else:
#             flat_w = torch.empty_like(self.flat_w).requires_grad_()
#         with torch.no_grad():
#             with self.unflatten_weight(flat_w):
#                 init_weights(self, init_type = 'xavier', init_param = 1.0)
#         return flat_w

# # 下面是你之前写的残差块和ResNet18结构，改为继承ReparamModule即可
# class ResidualBlock_18(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock_18, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )

#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet_18_reparam(ReparamModule):  # 继承 ReparamModule
#     def __init__(self, ResidualBlock, num_classes=10):
#         super(ResNet_18_reparam, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


# def ResNet18_reparam():
#     return ResNet_18_reparam(ResidualBlock_18)


# # #resnet18
# # class ResidualBlock_18(nn.Module):
# #     def __init__(self, inchannel, outchannel, stride=1):
# #         super(ResidualBlock_18, self).__init__()
# #         self.left = nn.Sequential(
# #             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
# #             nn.BatchNorm2d(outchannel),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
# #             nn.BatchNorm2d(outchannel)
# #         )
# #         self.shortcut = nn.Sequential()
# #         if stride != 1 or inchannel != outchannel:
# #             self.shortcut = nn.Sequential(
# #                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
# #                 nn.BatchNorm2d(outchannel)
# #             )
# #
# #     def forward(self, x):
# #         out = self.left(x)
# #         out += self.shortcut(x)
# #         out = F.relu(out)
# #         return out


# class ResNet_18(nn.Module):
#     def __init__(self, ResidualBlock, num_classes=10):
#         super(ResNet_18, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# def ResNet18():
#     return ResNet_18(ResidualBlock_18)

#resnet50和resnet101
class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=110):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = F.avg_pool2d(output,4)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output



def resnet50():

    return ResNet(BottleNeck, [3, 4, 6, 3])
#
# def resnet101():
#
#     return ResNet(BottleNeck, [3, 4, 23, 3])