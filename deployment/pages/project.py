import os
import time
import datetime
import sys
import numpy as np
import streamlit as st
import cv2
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils

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

generator = GeneratorResNet(3,3)
generator.load_state_dict(torch.load("./assets/generator_resnet.pth", map_location=torch.device('cpu')))

transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((256, 256), Image.BICUBIC),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def generate_image(uploaded_image):
    # Mengubah uploaded_image menjadi PIL Image
    pil_image = Image.open(uploaded_image).convert("RGB")
    
    # Transformasi menjadi tensor
    input_image = transform(pil_image).unsqueeze(0)

    # Generate image using the generator
    generated_image = generator(input_image)
    generated_image = generated_image.squeeze().permute(1, 2, 0)
    generated_image = torch.clamp(generated_image, 0, 1)
    generated_image = (generated_image * 255).to(torch.uint8)
    generated_image = generated_image.numpy()

    return generated_image

row1_col1, _ = st.columns(spec=2, gap="small")

tab1, tab2 = st.tabs(["Project", "Gallery"])

logo_tim = Image.open("./assets/Logo_Tim.png")
logo_sc = Image.open("./assets/[Logo] Startup Campus.png")
logo_km = Image.open("./assets/[Logo] Kampus Merdeka.png")

st.sidebar.image(logo_tim)
st.sidebar.image(logo_sc, caption="Startup Campus")
st.sidebar.image(logo_km, caption="Kampus Merdeka")

with row1_col1:
    st.title("VisionGen")

with tab1:
    st.subheader("project")
    uploaded_image = st.file_uploader("Uploade your image here ...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        st.write("Input")
        st.image(uploaded_image, width=128)
        
        if st.button("Generate"):
            generated_image = generate_image(uploaded_image)
            st.write("Output")
            st.image(generated_image, width=128)

with tab2:
    st.subheader("Gallery")
    for i in range(16):
        image_generate = Image.open(f"./assets/generated_image_{i+1}.jpg")
        st.image(image_generate)
    
