import torch
from torchvision import transforms
class Unet(torch.nn.Module):

    def __init__(
        self,
        encoder_pairs:int=4,
        initial_features:int=32,
        features_expanded:int=2,
        input_channels:int=1,
        conv_kern_size:int=3,
        trans_kern_size:int=2,
        pool_conv_kern_size:int=2,
        output_channels:int = 1,
        strides:int = 1,
        final_activation = torch.nn.Sigmoid
        ):

        super(Unet, self).__init__()
        #Don't know how Unet handels pretrained nets
        # if type(pretrained_net) is type(None):
        #     pass
        # else:
        #     self.pretrained_net = pretrained_net
        if (conv_kern_size%2 == 0 or conv_kern_size < 3):
            raise ValueError('kernal size must be odd and greator or equal to 3')
        else:
            self.conv_kernal_size=conv_kern_size

        self.padding = int((conv_kern_size-1)/2)
        if features_expanded<=1:
            raise ValueError('You cant multiply the number of channels by less than 1.')
        self.encoder_pairs = encoder_pairs
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(encoder_pairs+1):
            if i ==0:
                conv_input_channels = input_channels
                conv_output_channels = input_channels*initial_features
            else:
                conv_input_channels = conv_output_channels
                conv_output_channels = conv_input_channels*features_expanded
            self.encoder_layers.append(
                torch.nn.Conv2d(
                    conv_input_channels,
                    conv_output_channels,
                    conv_kern_size,
                    stride= strides,
                    padding = self.padding,
                )
            )
            self.encoder_layers.append(
                torch.nn.ReLU(inplace=True)
            )
            self.encoder_layers.append(
                torch.nn.Conv2d(
                    conv_output_channels,
                    conv_output_channels,
                    conv_kern_size,
                    stride= strides,
                    padding = self.padding,
                )
            )
            self.encoder_layers.append(
                torch.nn.ReLU(inplace=True)
            )
            if i != encoder_pairs:
                self.encoder_layers.append(
                    torch.nn.MaxPool2d(pool_conv_kern_size)
                )
        for i in range(encoder_pairs):
            conv_input_channels = conv_output_channels
            conv_output_channels = conv_input_channels//features_expanded
            self.decoder_layers.append(torch.nn.ConvTranspose2d(
                conv_input_channels,
                conv_output_channels,
                trans_kern_size,
                stride=2,
                padding=0,
                output_padding=0
            ))
            #Here, the skip connections are incorperated in the foward function
            self.decoder_layers.append(
                torch.nn.Conv2d(
                    conv_input_channels,
                    conv_output_channels,
                    kernel_size=conv_kern_size,
                    stride= strides,
                    padding = self.padding,
                )
            )
            self.decoder_layers.append(torch.nn.ReLU(inplace=True))
            self.decoder_layers.append(
                torch.nn.Conv2d(
                    conv_output_channels,
                    conv_output_channels,
                    kernel_size=conv_kern_size,
                    stride= strides,
                    padding = self.padding,
                )
            )
            self.decoder_layers.append(torch.nn.ReLU(inplace=True))
        self.final_conv = torch.nn.Conv2d(conv_output_channels, output_channels, conv_kern_size, padding=self.padding)
        self.final_activation = final_activation()
    
    def forward(self,x):
        if hasattr(self,'pretrained_net'):
            x = self.pretrained_net(x)
        skip_connections = []
        for layer in self.encoder_layers:
            if isinstance(layer,torch.nn.MaxPool2d):
                skip_connections.append(x)
            x = layer(x)
        i=1
        for layer in self.decoder_layers:
            x = layer(x)
            if isinstance(layer,torch.nn.ConvTranspose2d):
                x = torch.cat((
                    x,
                    transforms.functional.resized_crop(
                        skip_connections[self.encoder_pairs-i],
                        top=0,
                        left=0,
                        height=x.shape[-2],
                        width=x.shape[-1],
                        size=[x.shape[-2],
                        x.shape[-1]]
                    )
                ),
                dim=1)
                i+=1
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x
