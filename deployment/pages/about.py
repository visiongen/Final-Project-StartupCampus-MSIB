import streamlit as st
import numpy as np
import cv2
from PIL import Image

row1_col1, _ = st.columns(spec=2, gap='small')
row2_col1, _ = st.columns(spec=2)
row, _ = st.columns(spec=2)
row4_col1, row4_col2, row4_col3, row4_col4 = st.columns(spec=4, gap='small')
row5_col1, row5_col2, row5_col3 = st.columns(spec=3, gap="small")
row_facil_mentor, _ = st.columns(spec=2)
row3_col1, row3_col2 = st.columns(spec=2, gap='small')


logo_tim = Image.open("./assets/Logo_Tim.png")
logo_sc = Image.open("./assets/[Logo] Startup Campus.png")
logo_km = Image.open("./assets/[Logo] Kampus Merdeka.png")
foto_arsyad = Image.open("./assets/arsyad.jpg")
foto_dhifa = Image.open("./assets/dhifa.jpeg")
foto_pinka = Image.open("./assets/pinka.jpg")
foto_sultan = Image.open("./assets/sultan.jpg")
foto_abel = Image.open("./assets/abel.jpg")
foto_anan = Image.open("./assets/anan.jpg")
foto_mita = Image.open("./assets/mita.jpg")
foto_facil = Image.open("./assets/[Facil AI] Ni Luh Nitya Ayu Laksmi.jpeg")
foto_mentor = Image.open("./assets/[Mentor AI] Nicholas Dominic.jpeg")

with row1_col1:
    st.title("VisionGen")

st.sidebar.image(logo_tim, caption="VisionGen")
st.sidebar.image(logo_sc, caption="Startup Campus")
st.sidebar.image(logo_km, caption="Kampus Merdeka")


with row2_col1:
    st.header("About VisionGen")
    st.write("""
    VisionGen merupakan tim final project dari program studi independen kampus merdeka di startup campus trek artificial intelligence.
    """)

with row:
    st.header("Anggota")

with row4_col1:
    st.image(foto_arsyad)
    st.write("Muhammad Arsyad | Ketua")
    st.write("[linkedin](https://www.linkedin.com/in/muhammad-arsyad-59865120a/)")
    st.write("[Github](https://github.com/muharsyad)")
    
with row4_col2:
    st.image(foto_dhifa)
    st.write("Maulidia Nadhifa Aulia Shalsabila | Anggota")
    st.write("[linkedin](www.linkedin.com/in/maulidia-nadhifa-aulia-shalsabila-4978a9292)")
    st.write("[Github](https://github.com/nadhifaas)")

with row4_col3:
    st.image(foto_pinka)
    st.write("Pinka Ananda | Anggota")
    st.write("[linkedin](https://www.linkedin.com/in/pinka-ananda)")
    st.write("[Github](https://github.com/pinkaananda)")

with row4_col4:
    st.image(foto_sultan)
    st.write("Sultan Fahrezy Syahdwinata Nugraha | Anggota")
    st.write("[linkedin](https://www.linkedin.com/in/sultanconnect/)")
    st.write("[Github](https://github.com/sultanfsn)")

with row5_col1:
    st.image(foto_abel)
    st.write("Abel Yehud Silalahi | Anggota")
    st.write("[linkedin](https://id.linkedin.com/in/abel-yehud-silalahi-b18684228)")
    st.write("[Github](https://github.com/abelsilalahii)")

with row5_col2:
    st.image(foto_anan)
    st.write("Anandhita Ganang Alimana | Anggota")
    st.write("[linkedin](https://www.linkedin.com/in/anandhita-ganang-768a75219/)")
    st.write("[Github](https://github.com/alimanaanan)")

with row5_col3:
    st.image(foto_mita)
    st.write("Sari Mita Dewi | Anggota")
    st.write("[linkedin](https://www.linkedin.com/in/sari-mita-dewi-48a0a0292)")
    st.write("[Github](https://github.com)")

with row_facil_mentor:
    st.header("Facilitator & Supervisor")

with row3_col1:
    st.image(foto_facil, width=200)
    st.write("Ni Luh Nitya Ayu Laksmi | Facilitator")
    st.write("[Linkedin](https://linkedin.com)")

with row3_col2:
    st.image(foto_mentor, width=200)
    st.write("Nicholas Dominic | Supervisor")
    st.write("[linkedin](https://linkedin.com/in/nicholas-dominic)")
    st.write("[github](https://github.com/NicholasDominic)")
