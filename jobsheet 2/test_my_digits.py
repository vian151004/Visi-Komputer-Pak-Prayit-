# ===== TEST DENGAN GAMBAR SENDIRI =====
from tensorflow.keras import models
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os

# Load model CNN
model = models.load_model('model_cnn.h5')
print("✅ Model CNN loaded!")

def preprocess_custom_image(image_path):
    """Preprocess gambar tulisan tangan sendiri"""
    img = Image.open(image_path).convert('L')
    img = ImageOps.autocontrast(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# Test dengan gambar Anda
image_files = ['7.jpg']

print("\n🧪 Testing dengan gambar sendiri:")
for img_file in image_files:
    if os.path.exists(img_file):
        try:
            processed_img = preprocess_custom_image(img_file)
            prediction = model.predict(processed_img, verbose=0)
            pred_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            print(f"📸 {img_file} → Predicted: {pred_digit} (Confidence: {confidence:.3f})")
            
            # Tampilkan gambar
            plt.figure(figsize=(3, 3))
            plt.imshow(Image.open(img_file), cmap='gray')
            plt.title(f'Predicted: {pred_digit} ({confidence:.3f})')
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    else:
        print(f"⚠️  File {img_file} tidak ditemukan")