import torch
from torchvision import transforms
class Unet(torch.nn.Module):
    def __init__(
        self,
        encoder_pairs: int = 4,
        initial_features: int = 32,
        input_channels: int = 1,
        conv_kern_size: int = 3,
        output_channels: int = 1,
        strides: int = 1,
        final_activation = torch.nn.Sigmoid
    ):
        super(Unet, self).__init__()
        self.encoder_pairs = encoder_pairs
        self.initial_features = initial_features
        self.input_channels = input_channels
        self.conv_kern_size = conv_kern_size
        self.output_channels = output_channels
        self.strides = strides
        self.final_activation = final_activation
        self.validate_parameters(conv_kern_size)
        self.padding = self.calculate_padding(conv_kern_size)
        # encoder layers, conv_input_channels and conv_output_channels are then passed into the decoder layers
        self.encoder_layers, conv_output_channels = self.create_encoder_layers()
        # decoder layers
        self.decoder_layers, conv_output_channels = self.create_decoder_layers(conv_output_channels)
        self.final_conv = torch.nn.Conv2d(conv_output_channels, output_channels,self.conv_kern_size, padding=self.padding)
        self.final_activation = final_activation()
    
    def validate_parameters(self, conv_kern_size):
        if conv_kern_size % 2 == 0 or conv_kern_size < 3:
            raise ValueError('Kernel size must be odd and greater or equal to 3')
        
    def calculate_padding(self, conv_kern_size):
        return int((conv_kern_size - 1) / 2)

    def create_encoder_layers(self):
        layers = torch.nn.ModuleList()
        for i in range(self.encoder_pairs + 1):
            conv_input_channels = self.input_channels if i==0 else conv_output_channels
            conv_output_channels = conv_input_channels * self.initial_features if i == 0 else conv_input_channels * 2
            
            layers.append(torch.nn.Conv2d(
                conv_input_channels, conv_output_channels, self.conv_kern_size, stride=self.strides, padding=self.padding
            ))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(
                conv_output_channels, conv_output_channels, self.conv_kern_size, stride=self.strides, padding=self.padding
            ))
            layers.append(torch.nn.ReLU(inplace=True))
            if i != self.encoder_pairs:
                layers.append(torch.nn.MaxPool2d(2))
        return layers, conv_output_channels
    
    def create_decoder_layers(self,conv_output_channels):
        layers = torch.nn.ModuleList()
        for i in range(self.encoder_pairs):
            conv_input_channels = conv_output_channels
            conv_output_channels = conv_input_channels // 2
            
            layers.append(torch.nn.ConvTranspose2d(
                conv_input_channels, conv_input_channels//2, 2, stride=2, padding=0, output_padding=0
            ))
            #The channels are doubled here by concatenating in the forward pass
            layers.append(torch.nn.Conv2d(
                conv_input_channels, conv_output_channels, self.conv_kern_size, stride=self.strides, padding=self.padding
            ))
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Conv2d(
                conv_output_channels, conv_output_channels, self.conv_kern_size, stride=self.strides, padding=self.padding
            ))
            layers.append(torch.nn.ReLU(inplace=True))
        return layers,conv_output_channels
    
    def forward(self, x):
        if hasattr(self,'pretrained_net'):
            x = self.pretrained_net(x)
        x, skip_connections = self.encoder_forward(x)
        x = self.decoder_forward(x, skip_connections=skip_connections)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def encoder_forward(self,x):
        skip_connections = []
        for layer in self.encoder_layers:
            if isinstance(layer,torch.nn.MaxPool2d):
                skip_connections.append(x)
            x = layer(x)
        return x, skip_connections
    
    def decoder_forward(self, x, skip_connections):
        i = 1
        for layer in self.decoder_layers:
            x = layer(x)
            if isinstance(layer,torch.nn.ConvTranspose2d):
                x = torch.cat((
                    x,
                    transforms.functional.resized_crop(
                        skip_connections.pop(),
                        top=0,
                        left=0,
                        height=x.shape[-2],
                        width=x.shape[-1],
                        size=[x.shape[-2],
                        x.shape[-1]]
                    )
                ),
                dim=-3)
                i+=1
        return x

