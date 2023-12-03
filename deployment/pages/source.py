import streamlit as st
from PIL import Image

st.set_page_config(layout = 'wide', initial_sidebar_state= 'expanded')

image = Image.open("./assets/Logo_Tim.png")

st.sidebar.image(image,caption='Vision Gen', use_column_width=True )

st.subheader('Baseline Code (Using U-Net)')

code = '''
class UNetDown(nn.Module):
  def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
    super(UNetDown, self).__init__()
    layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
    if normalize:
      layers.append(nn.InstanceNorm2d(out_size))
    layers.append(nn.LeakyReLU(0.2))
    if dropout:
      layers.append(nn.Dropout(dropout))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

class UNetUp(nn.Module):
  def __init__(self, in_size, out_size, dropout=0.0):
    super(UNetUp, self).__init__()
    layers = [
        nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(out_size),
        nn.ReLU(inplace=True),
    ]
    if dropout:
      layers.append(nn.Dropout(dropout))

    self.model = nn.Sequential(*layers)

  def forward(self, x, skip_input):
    x = self.model(x)
    x = torch.cat((x, skip_input), 1)

    return x

class GeneratorUNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3):
    super(GeneratorUNet, self).__init__()

    self.down1 = UNetDown(in_channels, 64, normalize=False)
    self.down2 = UNetDown(64, 128)
    self.down3 = UNetDown(128, 256)
    self.down4 = UNetDown(256, 512, dropout=0.5)
    self.down5 = UNetDown(512, 512, dropout=0.5)
    self.down6 = UNetDown(512, 512, dropout=0.5)
    self.down7 = UNetDown(512, 512, dropout=0.5)
    self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

    self.up1 = UNetUp(512, 512, dropout=0.5)
    self.up2 = UNetUp(1024, 512, dropout=0.5)
    self.up3 = UNetUp(1024, 512, dropout=0.5)
    self.up4 = UNetUp(1024, 512, dropout=0.5)
    self.up5 = UNetUp(1024, 256)
    self.up6 = UNetUp(512, 128)
    self.up7 = UNetUp(256, 64)

    self.final = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.ZeroPad2d((1, 0, 1, 0)),
        nn.Conv2d(128, out_channels, 4, padding=1),
        nn.Tanh(),
    )

  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    d6 = self.down6(d5)
    d7 = self.down7(d6)
    d8 = self.down8(d7)
    u1 = self.up1(d8, d7)
    u2 = self.up2(u1, d6)
    u3 = self.up3(u2, d5)
    u4 = self.up4(u3, d4)
    u5 = self.up5(u4, d3)
    u6 = self.up6(u5, d2)
    u7 = self.up7(u6, d1)

    return self.final(u7)


'''
st.code(code, language='python')

st.subheader('Modified Architecture (Using ResNet)')

code = '''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(256, 256) for _ in range(num_residual_blocks)]
        )

        # Upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.downsampling(x)
        x = self.residual_blocks(x)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return torch.tanh(x)

'''
st.code(code, language = 'python')

st.subheader("Source Code Link:")
st.write("[link]https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py")
st.write("[link]https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/ResNet_pix2pix.py")

st.sidebar.success('Select a page above.')