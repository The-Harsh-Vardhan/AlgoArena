# Quick test script to verify the setup works
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Quick test of Fashion-MNIST loading
print("\n🔄 Testing Fashion-MNIST loading...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(f"✅ Fashion-MNIST loaded successfully!")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

print("\n🎯 Setup verification complete! Ready to run the notebook.")
