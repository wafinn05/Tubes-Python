import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "heartAttack.csv"
data = pd.read_csv(file_path)
print(data)

keterangan = {
    "age": "Usia",
    "sex": "Jenis kelamin",
    "cp": "Tipe nyeri dada",
    "trtbps": "Tekanan darah saat istirahat",
    "chol": "Kadar kolesterol",
    "fbs": "Gula darah",
    "rest_ecg": "Hasil EKG saat istirahat",
    "thalachh": "Detak jantung maksimum",
    "exang": "Angina akibat olahraga",
    "ca": "Jumlah pembuluh darah",
    "output": "Risiko serangan jantung"
}

def tampil_dataset():
    print("\nKETERANGAN KOLOM")
    for kolom in data.columns:
        if kolom in keterangan:
            print(kolom, ":", keterangan[kolom])
        else:
            print(kolom, ": Tidak ada deskripsi")

    print("\n5 DATA TERATAS")
    print(data.head())

    print("\n5 DATA TERAKHIR")
    print(data.tail())

def visualisasi_data():
    print("\nVISUALISASI DATA")

    plt.figure(figsize=(6,4))
    sns.countplot(x="output", data=data)
    plt.title("Distribusi Risiko Serangan Jantung")
    plt.xlabel("Risiko (0 = Tidak, 1 = Ya)")
    plt.ylabel("Jumlah Pasien")
    print("\n") 
    plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(data["age"], bins=10)
    plt.title("Distribusi Usia Pasien")
    plt.xlabel("Usia")
    plt.ylabel("Jumlah")
    print("\n")
    plt.show()

    plt.figure(figsize=(6,4))
    sns.boxplot(x="output", y="age", data=data)
    plt.title("Hubungan Usia dan Risiko Serangan Jantung")
    plt.xlabel("Risiko")
    plt.ylabel("Usia")
    print("\n")
    plt.show()
model = None

def machine_learning():
    global model

    fitur = ["age", "trtbps", "chol", "thalachh"]
    X = data[fitur]
    y = data["output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    akurasi = accuracy_score(y_test, model.predict(X_test))

    print("\nMODEL BERHASIL DILATIH")
    print("Akurasi:", round(akurasi * 100, 2), "%")

def prediksi_data():
    global model

    if model is None:
        print("Model belum dilatih")
        print("Jalankan menu Machine Learning terlebih dahulu")
        return

    print("\nMASUKKAN DATA PASIEN")

    umur = int(input("Umur pasien: "))
    tekanan = int(input("Tekanan darah: "))
    kolesterol = int(input("Kolesterol: "))
    detak = int(input("Detak jantung maksimum: "))

    data_input = [[umur, tekanan, kolesterol, detak]]
    hasil = model.predict(data_input)

    print("\nHASIL PREDIKSI")
    if hasil[0] == 1:
        print("Pasien BERISIKO serangan jantung")
    else:
        print("Pasien TIDAK berisiko serangan jantung")

while True:
    print("\nMENU PROGRAM")
    print("1. Tampilkan Dataset")
    print("2. Visualisasi Data")
    print("3. Machine Learning")
    print("4. Prediksi Data Baru")
    print("0. Keluar")

    pilihan = input("Pilih menu: ")

    if pilihan == "1":
        tampil_dataset()
    elif pilihan == "2":
        visualisasi_data()
    elif pilihan == "3":
        machine_learning()
    elif pilihan == "4":
        prediksi_data()
    elif pilihan == "0":
        print("Program selesai")
        break
    else:
        print("Pilihan tidak valid")