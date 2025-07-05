import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(df_path):
    """
    Melakukan pra-pemrosesan data kualitas udara.

    Args:
        df_path (str): Jalur ke file CSV dataset.

    Returns:
        pandas.DataFrame: DataFrame yang sudah diproses.
        sklearn.preprocessing.StandardScaler: Objek scaler yang sudah fit.
        dict: Kamus berisi objek LabelEncoder yang sudah fit untuk setiap kolom kategorikal.
    """

    df = pd.read_csv(df_path)

    # 1. Konversi Kolom 'tanggal' ke Datetime
    df['tanggal'] = pd.to_datetime(df['tanggal'])

    # Definisikan ulang kolom numerik setelah menghapus 'bulan' dan 'nama_bulan' (jika ada)
    if 'bulan' in df.columns:
        df.drop(columns=['bulan', 'nama_bulan'], inplace=True)


    # 2. Penanganan Missing Values
    # Kolom numerik untuk interpolasi
    numerical_cols_for_interpolation = ['pm10', 'so2', 'co', 'o3', 'no2', 'max']
    for col in numerical_cols_for_interpolation:
        if col in df.columns:
            df[col] = df[col].interpolate(method='linear')

    # Mengisi missing value di 'critical' dengan modus
    if 'critical' in df.columns:
        # Mengubah penggunaan inplace=True sesuai saran FutureWarning
        df['critical'] = df['critical'].fillna(df['critical'].mode()[0])

    # Menghapus kolom 'pm25' jika ada dan memiliki missing values yang signifikan
    if 'pm25' in df.columns:
        df.drop(columns=['pm25'], inplace=True)

    # Update daftar kolom numerik setelah drop 'pm25'
    current_numerical_cols = ['pm10', 'so2', 'co', 'o3', 'no2', 'max']
    current_numerical_cols = [col for col in current_numerical_cols if col in df.columns]

    # 3. Penanganan Outlier (IQR) - diulang dua kali
    # Iterasi 1
    Q1 = df[current_numerical_cols].quantile(0.25)
    Q3 = df[current_numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[~((df[current_numerical_cols] < lower_bound) | (df[current_numerical_cols] > upper_bound)).any(axis=1)].copy()

    # Iterasi 2 - untuk kolom yang mungkin masih memiliki outlier setelah iterasi pertama
    numerical_cols_for_second_iqr = ['pm10','so2','o3', 'no2', 'max']
    numerical_cols_for_second_iqr = [col for col in numerical_cols_for_second_iqr if col in df_cleaned.columns] 

    Q1_2 = df_cleaned[numerical_cols_for_second_iqr].quantile(0.25)
    Q3_2 = df_cleaned[numerical_cols_for_second_iqr].quantile(0.75)
    # Perbaikan: Hitung IQR_2 dengan benar
    IQR_2 = Q3_2 - Q1_2 
    lower_bound_2 = Q1_2 - 1.5 * IQR_2
    upper_bound_2 = Q3_2 + 1.5 * IQR_2
    df_cleaned = df_cleaned[~((df_cleaned[numerical_cols_for_second_iqr] < lower_bound_2) | (df_cleaned[numerical_cols_for_second_iqr] > upper_bound_2)).any(axis=1)].copy()

    # 4. Label Encoding
    categorical_cols = ['stasiun', 'critical', 'categori']
    label_encoders = {}
    for col in categorical_cols:
        if col in df_cleaned.columns:
            le = LabelEncoder()
            df_cleaned[col] = le.fit_transform(df_cleaned[col])
            label_encoders[col] = le

    # 5. Normalisasi Fitur (Feature Scaling)
    scaler = StandardScaler()
    df_scaled = df_cleaned.copy()
    
    # Kolom yang akan diskalakan adalah semua kolom numerik kecuali 'tanggal'
    columns_to_scale = df_cleaned.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    if 'tanggal' in columns_to_scale:
        columns_to_scale.remove('tanggal')

    df_scaled[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

    return df_scaled, scaler, label_encoders

# Set output directory
output_dir = 'preprocessing/ispu_preprocessing'
os.makedirs(output_dir, exist_ok=True) 

# Contoh penggunaan:
df_processed, fitted_scaler, fitted_label_encoders = preprocess_data('ispu_dki_all.csv')

# Simpan DataFrame yang sudah diproses ke CSV di dalam output_dir
df_processed.to_csv(os.path.join(output_dir, 'polutan_processed.csv'), index=False)

# Simpan scaler dan label encoders untuk penggunaan di masa mendatang (misalnya, untuk prediksi)
joblib.dump(fitted_scaler, os.path.join(output_dir, 'scaler.pkl'))
joblib.dump(fitted_label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))

print(f"Data preprocessing selesai. Data yang sudah diproses disimpan ke '{output_dir}/polutan_processed.csv'.")
print(f"Scaler disimpan ke '{output_dir}/scaler.pkl' dan Label Encoders ke '{output_dir}/label_encoders.pkl'.")
print("\nInfo DataFrame setelah preprocessing:")
df_processed.info()
print("\nHead DataFrame setelah preprocessing:")
print(df_processed.head())
