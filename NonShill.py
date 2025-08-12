import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib

def calculate_eccentricity(contour):
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis ** 2 / major_axis ** 2))
        return eccentricity
    else:
        return None

def calculate_metric(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        metric = (4 * np.pi * area) / (perimeter ** 2)
        return metric
    else:
        return None

def extract(image_path):
    """Ekstraksi fitur dari gambar"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Tidak bisa membaca gambar: {image_path}")
        
        resized_image = cv2.resize(image, (300, 300))
        
        # Crop image
        height, width = resized_image.shape[:2]
        start_x, start_y = width // 4, height // 4
        end_x, end_y = start_x + (width // 2), start_y + (height // 2)
        cropped_image = resized_image[start_y:end_y, start_x:end_x]
        
        # Convert ke grayscale
        cropped_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Fitur Warna
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
        
        features = []
        
        # 1. Color Features
        # BGR features
        for channel in range(3):
            features.extend([
                np.mean(cropped_image[:, :, channel]),
                np.std(cropped_image[:, :, channel]),
                np.percentile(cropped_image[:, :, channel], 25),
                np.percentile(cropped_image[:, :, channel], 75)
            ])
        
        # HSV features
        for channel in range(3):
            features.extend([
                np.mean(hsv[:, :, channel]),
                np.std(hsv[:, :, channel])
            ])
        
        # LAB features
        for channel in range(3):
            features.extend([
                np.mean(lab[:, :, channel]),
                np.std(lab[:, :, channel])
            ])
        
        # Fitur Tekstur (GLCM)
        glcm = graycomatrix(cropped_gray, distances=[1], angles=[0], 
                           levels=256, symmetric=True, normed=True)
        #0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Fitur Bentuk
        _, otsu_threshold = cv2.threshold(cropped_gray, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu_threshold, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Analisis eccentricity dan metric
        eccentricities = []
        metrics = []
        
        for contour in contours:
            ecc = calculate_eccentricity(contour)
            metric = calculate_metric(contour)
            
            if ecc is not None:
                eccentricities.append(ecc)
            if metric is not None:
                metrics.append(metric)
        
        avg_eccentricity = np.mean(eccentricities) if eccentricities else 0
        avg_metric = np.mean(metrics) if metrics else 0
        
        # Gabungkan semua fitur
        features.extend([
            contrast, correlation, energy, homogeneity,
            avg_eccentricity, avg_metric
        ])
        
        return features
    
    except Exception as e:
        print(f"Error dalam ekstraksi fitur: {str(e)}")
        return None

def load_dataset(base_path):
    """Load dataset dari folder"""
    features = []
    labels = []
    processed_count = 0
    error_count = 0

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            print(f"Memproses kelas: {class_name}")
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        image_features = extract(image_path)
                        if image_features is not None:
                            features.append(image_features)
                            labels.append(class_name)
                            processed_count += 1
                            if processed_count % 10 == 0:
                                print(f"Berhasil memproses {processed_count} gambar")
                    except Exception as e:
                        print(f"Error memproses {image_path}: {str(e)}")

    print(f"Total errors encountered: {error_count}")
    return np.array(features), np.array(labels)

def analyze_distances(X_train, X_test, y_train, y_test):
    """Analisis performa berbagai metrik jarak"""
    metrics = ['manhattan', 'euclidean', 'minkowski']
    n_neighbors_range = range(1, 21)  # Coba rentang nilai k yang lebih luas
    
    results = {}
    best_score = -1
    best_config = None
    
    plt.figure(figsize=(15, 10))
    
    for metric in metrics:
        results[metric] = {'accuracy': []}
        
        for n in n_neighbors_range:
            print(f"\nAnalisis untuk {metric} distance dengan k={n}")
            
            if metric == 'minkowski':
                knn = KNeighborsClassifier(n_neighbors=n, metric=metric, p=3)
            else:
                knn = KNeighborsClassifier(n_neighbors=n, metric=metric)
            
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Accuracy: {accuracy:.4f}")
            
            results[metric]['accuracy'].append(accuracy)
            
            if accuracy > best_score:
                best_score = accuracy
                best_config = {
                    'metric': metric,
                    'n_neighbors': n,
                    'accuracy': accuracy
                }
    
    # Plot hasil
    plt.subplot(1, 1, 1)
    for metric in metrics:
        plt.plot(n_neighbors_range, results[metric]['accuracy'], 
                marker='o', label=f'{metric}')
    plt.title('Perbandingan Accuracy')
    plt.xlabel('Jumlah Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKonfigurasi Terbaik:")
    print(f"Metrik Jarak: {best_config['metric']}")
    print(f"Jumlah Neighbors: {best_config['n_neighbors']}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    
    return best_config

def train_and_evaluate_model():
    """Training dan evaluasi model"""
    dataset_path = r"D:\! Code\Klasifikasi\dataset\Local"
    
    print("Loading dataset...")

    features, labels = load_dataset(dataset_path)
    
    if len(features) == 0:
        print("Dataset kosong")
        return None, None, None, 0
    
    print(f"\nTotal data: {len(features)} samples")
    print(f"Jumlah fitur: {features.shape[1]}")
    print(f"Kelas unik: {set(labels)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Handling missing values
    imputer = SimpleImputer(strategy='median')
    features = imputer.fit_transform(features)
    
    # Normalisasi
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, test_size=0.4, random_state=42, stratify=labels_encoded
    )
    
    print("\nMenganalisis performa metrik jarak...")
    best_config = analyze_distances(X_train, X_test, y_train, y_test)
    
    # Train model final dengan konfigurasi terbaik
    if best_config['metric'] == 'minkowski':
        final_model = KNeighborsClassifier(
            n_neighbors=best_config['n_neighbors'],
            metric=best_config['metric'],
            p=3
        )
    else:
        final_model = KNeighborsClassifier(
            n_neighbors=best_config['n_neighbors'],
            metric=best_config['metric']
        )
    
    final_model.fit(X_train, y_train)
    
    # Evaluasi model
    y_pred = final_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    # Simpan model
    joblib.dump(final_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    return final_model, scaler, label_encoder, best_config['accuracy']

def predict_single_image(model, scaler, label_encoder, image_path):
    """Prediksi kelas untuk satu gambar"""
    try:
        features = extract(image_path)
        if features is None:
            raise ValueError("Gagal mengekstrak fitur")
        
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return label_encoder.inverse_transform(prediction)[0]
    
    except Exception as e:
        print(f"Error dalam prediksi: {str(e)}")
        return None

def validate_single_image(model_path, scaler_path, label_encoder_path, image_path):
    """Validasi gambar baru menggunakan model yang sudah dilatih."""
    try:
        # Muat model, scaler, dan label encoder
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        
        # Ekstraksi fitur dari gambar baru
        features = extract(image_path)
        if features is None:
            raise ValueError("Gagal mengekstrak fitur")
        
        # Normalisasi fitur
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Prediksi kelas
        prediction = model.predict(features_scaled)
        predicted_class = label_encoder.inverse_transform(prediction)[0]
        
        return predicted_class
    
    except Exception as e:
        print(f"Error dalam validasi gambar: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting training process...")
    model, scaler, label_encoder, accuracy = train_and_evaluate_model()
    
    if model is None:
        print("Training gagal. Program berhenti.")
        exit()
    
    print("\nModel siap untuk prediksi!")
    print(f"Accuracy model: {accuracy:.4f}")
    
    while True:
        root = Tk()
        root.withdraw()
        print("\nPilih opsi:")
        print("1. Prediksi gambar baru")
        print("2. Validasi gambar baru menggunakan model yang sudah disimpan")
        print("3. Keluar")
        choice = input("Masukkan pilihan (1/2/3): ").strip()
        
        if choice == '1':
            image_path = filedialog.askopenfilename(
                filetypes=[("Image Files", ".jpg;.png;*.jpeg")]
            )
            
            if not image_path:
                print("Tidak ada file yang dipilih.")
            else:
                prediction = predict_single_image(model, scaler, label_encoder, image_path)
                if prediction is not None:
                    print(f"\nHasil Prediksi: {prediction}")
        
        elif choice == '2':
            model_path = 'model.pkl'
            scaler_path = 'scaler.pkl'
            label_encoder_path = 'label_encoder.pkl'
            
            image_path = filedialog.askopenfilename(
                filetypes=[("Image Files", ".jpg;.png;*.jpeg")]
            )
            
            if not image_path:
                print("Tidak ada file yang dipilih.")
            else:
                predicted_class = validate_single_image(model_path, scaler_path, label_encoder_path, image_path)
                if predicted_class is not None:
                    print(f"\nHasil Validasi: Gambar termasuk dalam kelas '{predicted_class}'")
        
        elif choice == '3':
            print("Program selesai.")
            break
        
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")