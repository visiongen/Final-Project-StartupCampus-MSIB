import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(layout = 'wide', initial_sidebar_state= 'expanded')

image = Image.open("./assets/Logo_Tim.png")

st.sidebar.image(image,caption='Vision Gen', use_column_width=True )

st.title('Produsket')
st.subheader('Generate your Sketch Image into Real Image!')
st.sidebar.success('Select a page above.')
st.caption(
    '''In this digital era, around 70% of manual product design processing techniques are still 
    widely used by designers or companies, thus resulting in
    relatively expensive prices, long processing times and require extra 
    attention in the process. Produsket can help you to make experiment with color
    and different materials without having to create a physical prototype. We can help you to 
    make your own best product design! '''
)