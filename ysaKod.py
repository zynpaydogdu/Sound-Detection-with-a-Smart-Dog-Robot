#sıla
import librosa
import numpy as np
import os
import sounddevice as sd 
import wave
import pickle  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ffmpeg
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

def convert_m4a_to_wav(m4a_file, wav_file):
    try:
        ffmpeg.input(m4a_file).output(wav_file).overwrite_output().run()
        print(f"{m4a_file} başarıyla {wav_file} olarak dönüştürüldü.")
    except Exception as e:
        print(f"Error converting {m4a_file} to {wav_file}: {e}") 

def extract_features(filename, n_mfcc=20):
    try:
        y, sr = librosa.load(filename, sr=44100)
        y = y / np.max(np.abs(y)) 
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

    if y is None or len(y) == 0:
        print(f"Hata: {filename} yüklenemedi veya boş.")
        return None
    
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    except Exception as e:
        print(f"Özellik çıkarımı sırasında hata: {e}")
        return None
    
    try:
        features = np.hstack([ 
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1)
        ])
    except Exception as e:
        print(f"Özellikleri birleştirirken hata: {e}")
        return None
    
    return features

def augment_audio(y, sr):
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    y_slow = librosa.effects.time_stretch(y, rate=0.9)
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    return [y, y_fast, y_slow, y_noise]

def create_dataset(wav_folder, scaler=None):
    X, y = [], []
    class_counts = {}
    for label in ["0", "1", "2", "3", "4"]:
        folder_path = os.path.join(wav_folder, label)
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                filepath = os.path.join(folder_path, file)
                features = extract_features(filepath)
                if features is not None:
                    X.append(features)
                    y.append(int(label))
                    class_counts[label] = class_counts.get(label, 0) + 1
    
    print("Sınıf Dağılımı:", class_counts)
    X = np.array(X)
    if scaler:
        X = scaler.fit_transform(X)
    
    return X, np.array(y)

def convert_all_m4a_in_folder(folder_path):
    for label in ["0", "1", "2", "3", "4"]:
        label_folder = os.path.join(folder_path, label)
        if not os.path.exists(label_folder):
            continue

        for file in os.listdir(label_folder):
            if file.endswith(".m4a"):
                m4a_path = os.path.join(label_folder, file)
                wav_path = os.path.splitext(m4a_path)[0] + ".wav"
                convert_m4a_to_wav(m4a_path, wav_path)

def build_ann_model(input_shape, num_classes=5):
    input_layer = Input(shape=(input_shape,))

    x = Dense(256, activation='relu')(input_layer)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_wo_mic_device_index():
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "wo mic" in device['name'].lower() and device['max_input_channels'] > 0:
            return i
    return None

def check_wo_mic_connection():
    device_index = get_wo_mic_device_index()
    if device_index is None:
        print("WO Mic mikrofonu bulunamadı. Lütfen mikrofonun bağlı olduğundan emin olun.")
        return False
    return True

def record_audio(filename="C:/Users/aysez/Desktop/deprem_kopek/real_time_audio.wav", duration=5, fs=44100):
    print(f"Ses kaydedilecek dosya: {filename}")
    if not check_wo_mic_connection():
        return
    print("Konuşmaya başlayın...")
    
    device_index = get_wo_mic_device_index()
    if device_index is None:
        print("WO Mic mikrofonu bulunamadı. Lütfen mikrofonun bağlı olduğundan emin olun.")
        return
    
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32, device=device_index)
        sd.wait()
        print("Ses kaydedildi.")

        recording = recording / np.max(np.abs(recording))

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
        print(f"Ses başarıyla {filename} dosyasına kaydedildi.")
    
    except Exception as e:
        print(f"Ses kaydetme hatası: {e}")

def real_time_classification():
    wav_file = "C:/Users/aysez/Desktop/deprem_kopek/real_time_audio.wav"
    record_audio(wav_file)

    if not os.path.exists(wav_file):
        print("Ses dosyası oluşturulamadı.")
        return

    features = extract_features(wav_file)
    if features is None:
        print("Özellik çıkarılamadı.")
        return

    features = scaler.transform([features])
    features = np.expand_dims(features, axis=-1)

    model = load_model("trained_model_ann.h5") 
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    
    komutlar = {
        0: "Çalışmayı Başladı",
        1: "Çalışmayı Sonlandırdı",
        2: "Aramalara Başladı",
        3: "Aramaları Bitirdi",
        4: "Durdu"
    }
    print(f"Köpek {komutlar.get(predicted_class, 'Tanımlanmadı')}!")

def build_cnn_model(input_shape, num_classes=5):
    input_layer = Input(shape=(input_shape, 1))

    x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

wav_folder = "C:/Users/aysez/Desktop/deprem_kopek/ses_datalari"
convert_all_m4a_in_folder(wav_folder)

scaler = StandardScaler()
X, y = create_dataset(wav_folder, scaler)

with open("features.pkl", "wb") as f:
    pickle.dump((X, y), f)

X = np.expand_dims(X, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

ann_model = build_ann_model(X.shape[1])
ann_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = ann_model.evaluate(X_test, y_test)
print(f"ANN Model doğruluk oranı: {accuracy * 100:.2f}%")
ann_model.save("trained_model_ann.h5")

cnn_model = build_cnn_model(X.shape[1])
cnn_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = cnn_model.evaluate(X_test, y_test)
print(f"CNN Model doğruluk oranı: {accuracy * 100:.2f}%")
cnn_model.save("trained_model_cnn.h5")

real_time_classification()