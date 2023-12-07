import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import (silhouette_score, accuracy_score)

# Read the data into a pandas DataFrame
df = pd.read_csv("./Survey.csv")

# Title
st.title("Survey Kepuasan Mahasiswa Universitas Airlangga Terhadap Fasilitas dan Pelayanan Kantin Kampus C")

# Sliders for user inputs
def convert_radio_choice(choice):
    choices_map = {
        'Tidak puas': 1,
        'Hampir tidak puas': 2,
        'Cukup': 3,
        'Cukup puas': 4,
        'Sangat puas': 5
    }
    return choices_map.get(choice, 0)

variasi = st.radio('Bagaimana menurut anda variasi makanan yang ada di kantin?',
                            ['Tidak puas', 'Hampir tidak puas', 'Cukup', 'Cukup puas', 'Sangat puas'], key='variasi')
variasi = convert_radio_choice(variasi)

harga = st.radio('Apakah harga makanan dan minuman di kantin sudah cukup terjangkau?',
                         ['Tidak puas', 'Hampir tidak puas', 'Cukup', 'Cukup puas', 'Sangat puas'], key='harga')
harga = convert_radio_choice(harga)

pembayaran = st.radio('Apakah anda puas dengan sistem pembayaran yang diterapkan di kantin?',
                            ['Tidak puas', 'Hampir tidak puas', 'Cukup', 'Cukup puas', 'Sangat puas'], key='pembayaran')
pembayaran = convert_radio_choice(pembayaran)

fasilitas = st.radio('Apakah anda puas dengan fasilitas yang disediakan di kantin?',
                         ['Tidak puas', 'Hampir tidak puas', 'Cukup', 'Cukup puas', 'Sangat puas'], key='fasilitas')
fasilitas = convert_radio_choice(fasilitas)

# Handling missing values
df = df.dropna()
df = df.fillna(df.mean())

threshold = 3
outliers_df = pd.DataFrame()

for col in df.columns:
    z_scores = np.abs(stats.zscore(df[col]))
    is_outlier = z_scores > threshold
    outliers_df[col] = is_outlier

outliers_summary = outliers_df.sum()

X = df.iloc[:, 1:10]

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah kluster optimal menggunakan Silhouette analysis
silhouette_scores = []
for i in range(2, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print(silhouette_scores)
# Plot grafik Silhouette analysis
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Analysis')
plt.xlabel('Jumlah kluster')
plt.ylabel('Silhouette Score')
plt.show()

# Pilih jumlah kluster berdasarkan Silhouette analysis
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2 

# Menerapkan algoritma K-Means
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled) + 1

# Menambahkan hasil kluster ke dalam DataFrame X
X['Cluster'] = df['Cluster']

# Classification and evaluation
x = df[['variasi', 'harga', 'pembayaran', 'fasilitas']]
y = df[['Cluster']]

# Initialize models
gaussian = GaussianNB()

x_1, x_2, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', random_state=0)
x_train = kbins.fit_transform(x_1)
x_test = kbins.transform(x_2)

# Naive Bayes
gaussian.fit(x_train, y_train)
y_predictionNB = gaussian.predict(x_test)
accuracy_nb = round(accuracy_score(y_test, y_predictionNB)* 100, 2)
acc_gaussianNB = round(gaussian.score(x_train, y_train)* 100, 2)

# Predict user satisfaction
user_data = [[variasi, harga, pembayaran, fasilitas]]
user_data = kbins.transform(user_data)

# Predictions
predictionNB = gaussian.predict(user_data)

# Streamlit app
if st.button("Predict Satisfaction"):
    st.subheader("Prediction Results")
    st.write("Anda Termasuk Pengguna Kantin dengan predikat PUAS" if predictionNB[0] == 2 else "Anda Termasuk Pengguna Kantin dengan predikat TIDAK PUAS silahkan hubungi nomor berikut 082111633000 untuk peningkatan layanan kantin dan beri masukan")