🧠 CIFAR-10 Image Classifier (TensorFlow + Streamlit)
📸 Overview

This project is a deep learning–based image classification app built with TensorFlow and Streamlit.
It uses the CIFAR-10 dataset to classify uploaded images into one of 10 categories:

✈️ Airplane, 🚗 Automobile, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐶 Dog, 🐸 Frog, 🐎 Horse, 🚢 Ship, 🚚 Truck

The model is trained using a Convolutional Neural Network (CNN) and deployed as an interactive Streamlit web app for real-time image recognition.

🚀 Features

🔥 TensorFlow CNN model trained on CIFAR-10

🖼️ Upload any image (.jpg, .png, .jpeg)

⚡ Real-time prediction and confidence bar chart

📊 Streamlit-powered user interface

💾 Model saved and loaded (.h5) for deployment


Image_Classifier/
│
├── train_model.py        # Builds, trains, and saves the CNN model
├── frontend.py           # Streamlit app for user interface
├── model_cifar10_v2.h5   # Saved model (generated after training)
├── README.md             # Project documentation
└── LICENSE               # Open-source license (MIT)

🧑‍💻 Author

Joel Zerubabel

❤️ Acknowledgements

TensorFlow

Streamlit

CIFAR-10 Dataset

