# Produsket : Transformasi Gambar Produk Dari Sketsa Dengan Cepat

## Project Description
Please describe your Startup Campus final project here. You may should your <b>model architecture</b> in JPEG or GIF.

## Contributor
| Full Name | Affiliation | Email | LinkedIn | Role |
| --- | --- | --- | --- | --- |
| Muhammad Arsyad | Universitas Sebelas Maret | muharsyad2201@gmail.com | [link](https://id.linkedin.com/in/muhammad-arsyad-59865120a) | Team Lead |
| Maulidia Nadhifa Aulia Shalsabila | Universitas Airlangga | maulidianadhifa@gmail.com | [link](https://linkedin.com/in/maulidia-nadhifa-aulia-shalsabila-4978a9292) | Team Member |
| Pinka Ananda | Universitas Lampung | pinkaananda@gmail.com | [link](https://www.linkedin.com/in/pinka-ananda) |Team Member |
| Sultan Fahrezy Syahdwinata Nugraha | Universitas Indonesia | sultan.fahrezy.sn@gmail.com | [link](https://www.linkedin.com/in/sultanconnect/) | Team Member |
| Abel Yehud Silalahi | Universitas Jendral Achmad Yani Yogyakarta | abelyehuds@gmail.com | [link](https://id.linkedin.com/in/abel-yehud-silalahi-b18684228) | Team Member |
| Anandhita Ganang Alimana | Universitas Indonesia | alimanaanandhita15@gmail.com | [link](https://www.linkedin.com/in/anandhita-ganang-768a75219/) | Team Member |
| Sari Mita Dewi | STMIK Insan Pembangunan | sarimitadewi10@gmail.com | [link](https://www.linkedin.com/in/sari-mita-dewi-48a0a0292) | Team Member |
| Nicholas Dominic | Startup Campus, AI Track | nic.dominic@icloud.com | [link](https://linkedin.com/in/nicholas-dominic) | Supervisor |

## Setup
### Prerequisite Packages (Dependencies)
- pandas==1.5.3
- numpy==1.23.5
- torch==2.1.0+cu118
- torchvision==0.16.0+cu118
- opencv-python==4.8.0
- Pillow==9.4.0
- matplotlib==3.7.1
- os
- math
- time
- itertools
- datetime
- sys
- argparse
- torchsummary
- google.colab.patches
- ...
- ...

### Environment
| | |
| --- | --- |
| CPU | Example: Apple M3 Pro 12-core CPU |
| GPU | Example: Nvidia A100 (x1) |
| ROM | Example: 1 TB SSD |
| RAM | Example: 36 GB |
| OS | Example: macOS Sonoma v14.1.1 |

## Dataset
Kami menggunakan data berupa gambar fashion yang tersedia dari [kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Dataset yang kami gunakan sebantak 7 kelas yaitu gambar kacamata sebanyak 1000 data, jam tangan sebanyak 2558 gambar, tas sebanyak 1000 gambar, bawahan sebanyak 790 gambar, atasan sebanyak 1330 gambar, sepatu sebanyak 1000 gambar dan sandal sebanyak 1876 gambar. Dengan total sebanyak 8744 gambar dimana kami bagi menjadi dataset untuk training sebesar 70% atau sebanyak 6691 gambar, lalau dataset test sebesar 20% atau sebanyak 1906 gambar, dan data validasi sebesar 10% atau sebanyak 957 gambar. Setelah itu, kami lakukan image processing untuk mendapatkan gambar sketsa dengan cara edge detection. Berikut link untuk dataset yang telah dilakukan image processing
- Link: [https://www.kaggle.com/datasets/arsyadmuhammad/edge-2-real-image/data](https://www.kaggle.com/datasets/arsyadmuhammad/edge-2-real-image/data)

berikut contoh data

![image](https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/input%20image.png?raw=true)

## Results
### Model Performance
Describe all results found in your final project experiments, including hyperparameters tuning and architecture modification performances. Put it into table format. Please show pictures (of model accuracy, loss, etc.) for more clarity.

#### 1. Metrics
Inform your model validation performances, as follows:
- For classification tasks, use **Precision and Recall**.
- For object detection tasks, use **Precision and Recall**. Additionaly, you may also use **Intersection over Union (IoU)**.
- For image retrieval tasks, use **Precision and Recall**.
- For optical character recognition (OCR) tasks, use **Word Error Rate (WER) and Character Error Rate (CER)**.
- For adversarial-based generative tasks, use **Peak Signal-to-Noise Ratio (PNSR)**. Additionally, for specific GAN tasks,
  - For single-image super resolution (SISR) tasks, use **Structural Similarity Index Measure (SSIM)**.
  - For conditional image-to-image translation tasks (e.g., Pix2Pix), use **Inception Score**.

Feel free to adjust the columns in the table below.

| model | epoch | learning_rate | batch_size | optimizer | val_loss | val_precision | val_recall | ... |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | 1000 |  0.0001 | 32 | Adam | 0.093 | 88.34% | 84.15% | ... |
| vit_l_32 | 2500 | 0.00001 | 128 | SGD | 0.041 | 90.19% | 87.55% | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | 

#### 2. Ablation Study
Any improvements or modifications of your base model, should be summarized in this table. Feel free to adjust the columns in the table below.

| model | layer_A | layer_B | layer_C | ... | top1_acc | top5_acc |
| --- | --- | --- | --- | --- | --- | --- |
| vit_b_16 | Conv(3x3, 64) x2 | Conv(3x3, 512) x3 | Conv(1x1, 2048) x3 | ... | 77.43% | 80.08% |
| vit_b_16 | Conv(3x3, 32) x3 | Conv(3x3, 128) x3 | Conv(1x1, 1028) x2 | ... | 72.11% | 76.84% |
| ... | ... | ... | ... | ... | ... | ... |

#### 3. Training/Validation Curve
Insert an image regarding your training and evaluation performances (especially their losses). The aim is to assess whether your model is fit, overfit, or underfit.

Berikut grafik discriminator loss dan generator loss

<img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/discriminator_loss.png?raw=true" width=50%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/generator_loss.png?raw=true" width=50%>
 
### Testing

Berikut hasil gambar yang telah digenerate, dimana pada baris pertama merupakan gambar input, baris kedua merupakan gambar asli, dan baris ketiga merupakan hasil generate

<img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_736.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_809.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_810.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_855.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_882.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_885.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_887.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_889.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_922.jpg?raw=true" width=10%><img src="https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/image_950.jpg?raw=true" width=10%>

### Deployment (Optional)
Pada deployment kami menggunakan streamlit, dapat diakses pada link berikut : [visiongen.streamlit.app](https://visiongen.streamlit.app).

berikut merupakan screenshot

![image](https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/deployment.jpeg?raw=true)

anda dapat memasukan gambar sketsa lalu mengenerate untuk memperoleh gambar desain nyata, seperti pada video di bawah ini:

![video](https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/assets/deployment.gif?raw=true)

## Supporting Documents
### Presentation Deck
- Link: [https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/presentasi.pdf](https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/presentasi.pdf)

### Business Model Canvas
- Link : [https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/bussines%20model%20canvas.pdf](https://github.com/visiongen/Final-Project-StartupCampus-MSIB/blob/main/bussines%20model%20canvas.pdf)

### Short Video
Provide a link to your short video, that should includes the project background and how it works.
- Link: https://...

## References
Provide all links that support this final project, i.e., papers, GitHub repositories, websites, etc.
- Link: [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- Link: https://...
- Link: https://...

## Additional Comments
Provide your team's additional comments or final remarks for this project. For example,
1. ...
2. ...
3. ...

## How to Cite
If you find this project useful, we'd grateful if you cite this repository:
```
@article{
...
}
```

## License
For academic and non-commercial use only.

## Acknowledgement
This project entitled <b>"Produsket : Transformasi Gambar Produk Dari Sketsa Dengan Cepat"</b> is supported and funded by Startup Campus Indonesia and Indonesian Ministry of Education and Culture through the "**Kampus Merdeka: Magang dan Studi Independen Bersertifikasi (MSIB)**" program.
