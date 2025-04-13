🐶 Sound Detection with a Smart Dog Robot
This project is a machine learning system that recognizes voice commands for a search-and-rescue dog. The dog detects and responds to predefined commands (start working, stop, start searching, stop searching).

🚀 Features
Automatically converts .m4a audio files to .wav format

Extracts audio features such as MFCC, mel, chroma, and spectral contrast

Records real-time audio using WO Mic

Classifies commands using ANN and CNN models

Predicts the dog’s command in real-time

🛠️ Technologies Used
Python & TensorFlow

Librosa & SoundDevice

Scikit-learn

FFmpeg

Keras

📁 Core Functions
python
Kopyala
Düzenle
# Convert audio files
convert_all_m4a_in_folder()

# Create dataset
create_dataset()

# Train ANN and CNN models
build_ann_model()
build_cnn_model()

# Real-time audio recording and classification
real_time_classification()
📌 Command Labels
Label	Meaning
0	Started Working
1	Stopped Working
2	Started Searching
3	Finished Searching
4	Stopped
🎙️ Note
For real-time voice recognition, WO Mic must be connected to both your computer and your phone. Otherwise, the system may not function properly. And also this project needs labeled audio data to train the models. 
