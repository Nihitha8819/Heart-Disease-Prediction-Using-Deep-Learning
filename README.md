# Heart-Disease-Prediction-Using-Deep-Learning

📌 Project Overview
This project builds an offline machine learning model to predict chronic heart disease using Keras (TensorFlow) and NumPy. Given patient health metrics, the model classifies whether a person has heart disease (binary outcome: disease/no disease).
🧠 Goal: Leverage AI in healthcare to assist in early risk screening for heart disease, a leading global cause of death.

✅ Key Features
🧪 Predicts presence/absence of heart disease from clinical data.

⚙️ Built with Python, Keras (TensorFlow), and NumPy.

📁 Includes full training and evaluation pipelines in Google Colab notebooks.

📉 Achieves ~80–90% test accuracy on unseen data.

🗂️ Files in the Repository
File	Description
heart_disease_prediction_training.ipynb	Notebook for data preprocessing, model training, and evaluation.
heart_model_testing.ipynb	Notebook to load the saved model and test it with new data.
README.md	Project documentation (you're reading it!).

🧬 Dataset Description
We used a version of the UCI Cleveland Heart Disease Dataset, containing ~300 patient records. Each record includes attributes like:
age: Age in years
sex: Gender (1 = male, 0 = female)
cp: Chest pain type
trestbps: Resting blood pressure
chol: Serum cholesterol
fbs: Fasting blood sugar (> 120 mg/dl)
restecg: Resting ECG results
thalach: Maximum heart rate achieved
exang: Exercise-induced angina
oldpeak: ST depression induced by exercise

➡️ The target label is binary:
0 = No disease
1 = Disease present

🧠 Model Architecture
The model is a feed-forward Artificial Neural Network (ANN) built with Keras' Sequential API.
Input Layer: Matches number of input features (e.g., 10)
Hidden Layer: Dense (5 neurons) with ReLU activation
Output Layer: Dense (1 neuron) with Sigmoid activation

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

🛠️ Training Pipeline (in training notebook)
1) Data Preprocessing
Standardization of numeric features
Label encoding of categorical values
Handling missing data (if any)
2) Splitting the Dataset
80/20 or 70/30 train-test split
Optionally, cross-validation
3) Model Training
model.fit() over 100 epochs
Batch size: 32
Accuracy and loss monitored
4) Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2%}")
📊 Key Results and Insights
Accuracy: ~80–90% on test data

Model Behavior: Balanced performance across classes (checked via confusion matrix)

Feature Trends: Higher risk observed with higher age, cholesterol, and BP—matching real-world medical insights

🔬 Note: While the model does not provide explicit feature importance, its behavior aligns with known risk factors for heart disease.

⚠️ Disclaimer
🚨 Not for Clinical Use:
This model is for educational and research purposes only. It is not certified for clinical or real-world deployment.
Accuracy may vary with different datasets or populations.
This tool is not a substitute for medical advice.
Always consult healthcare professionals for medical decisions.

