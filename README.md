# 🐱🐶 Cat and Dog Image Classification

A deep learning project that classifies images as either **cats** or **dogs** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## 📊 Dataset

- **Source**: [Kaggle - Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- Images split into training and validation sets.
- Folder structure:
- dataset/
├── train/
│ ├── cats/
│ └── dogs/
└── validation/
├── cats/
└── dogs/


---

## 🧠 Model Summary

- Input: 150x150 RGB images
- Architecture:
- 3 Conv2D + MaxPooling layers
- Flatten
- Dense (512) + ReLU
- Output: Sigmoid (binary classification)
- Loss: `binary_crossentropy`
- Optimizer: `adam`

---

## 📈 Training and Evaluation

- `model.fit()` used to train over multiple epochs
- Training and validation accuracy/loss visualized
- Model saved as `.keras` and `.h5` formats

---

## 🔍 Predicting New Images

```python
img = load_img('path_to_image.jpg', target_size=(150,150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
pred = model.predict(img_array)
print("Prediction:", "Dog" if pred > 0.5 else "Cat")

📦 Dependencies
TensorFlow / Keras

NumPy

Matplotlib

PIL (for image loading)

