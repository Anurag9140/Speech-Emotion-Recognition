import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Extract Features Function

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3.0, offset=0.5)

        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

        feature_vector = np.hstack((mfccs, chroma, mel))
        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Load TESS Dataset

def load_tess_data(tess_dir):
    features = []
    labels = []

    print(f"Loading TESS dataset from: {tess_dir}")
    for root, dirs, files in os.walk(tess_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                
                # Get emotion from parent folder name
                label_folder = os.path.basename(root).lower()

                if 'angry' in label_folder:
                    label = 'angry'
                elif 'disgust' in label_folder:
                    label = 'disgust'
                elif 'fear' in label_folder:
                    label = 'fearful'
                elif 'happy' in label_folder:
                    label = 'happy'
                elif 'neutral' in label_folder:
                    label = 'neutral'
                elif 'pleasant_surprise' in label_folder or 'pleasant_surprised' in label_folder:
                    label = 'surprised'
                elif 'sad' in label_folder:
                    label = 'sad'
                else:
                    print(f"⚠️ Skipping unmatched folder: {root}")
                    continue

                feature_vector = extract_features(file_path)
                if feature_vector is not None:
                    features.append(feature_vector)
                    labels.append(label)

    return np.array(features), np.array(labels)


# Train and Evaluate Model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=500, random_state=42)
    print("\nTraining the MLPClassifier model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return model


if __name__ == "__main__":
    tess_data_dir = 'data/TESS'

    X, y = load_tess_data(tess_data_dir)
    print(f"\nTotal Samples Loaded: {len(X)}")

    if len(X) == 0:
        print("❌ No samples loaded. Please check dataset folder structure.")
    else:
        model = train_model(X, y)

        
import joblib
joblib.dump(model, 'tess_ser_mlp_model.pkl')
