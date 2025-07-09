# 🧠 Next Word Prediction using LSTM on Hamlet Dataset

![GitHub repo size](https://img.shields.io/github/repo-size/sg2499/Hamlet-Next-Word-Prediction-LSTM)
![GitHub stars](https://img.shields.io/github/stars/sg2499/Hamlet-Next-Word-Prediction-LSTM?style=social)
![Last Commit](https://img.shields.io/github/last-commit/sg2499/Hamlet-Next-Word-Prediction-LSTM)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-orange)

This repository presents an end-to-end deep learning pipeline that performs **next word prediction** using **LSTM** on a dataset derived from *The Tragedie of Hamlet* by William Shakespeare. The model learns to predict the next likely word given an input phrase using sequential modeling.

---

## 📁 Project Structure

```
Hamlet-Next-Word-Prediction-LSTM/
├── app.py                      # Streamlit web app interface
├── hamlet.txt                  # Text corpus from Hamlet
├── Next_Word_Predictor.h5     # Trained LSTM model
├── Tokenizer.pkl              # Saved tokenizer object
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── Next Word Prediction using LSTM RNN.ipynb  # Training notebook
```

---

## 🧠 Model Summary

- **Dataset:** Shakespeare's *Hamlet* (raw text corpus)
- **Preprocessing:**
  - Lowercasing, punctuation removal
  - Tokenization and sequence padding
- **Model Architecture:**
  - Embedding Layer
  - LSTM Layer
  - Dense (Softmax Output) Layer
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Goal:** Predict the next word in a sentence from the learned Shakespearean style

> ⚠️ **Note:** Achieving very high accuracy is inherently challenging due to the poetic and non-standard nature of the Hamlet text.

---

## 🌐 Web App – Streamlit Interface

The deployed Streamlit web app allows users to:
- Type a sequence of words from Hamlet or any text
- Predict the **next word** using the trained LSTM model
- Enjoy a simple and clean UI for literary language modeling

### 🖥️ Screenshot
> *(Add app UI screenshot here if available)*

---

## 💾 Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/sg2499/Hamlet-Next-Word-Prediction-LSTM.git
cd Hamlet-Next-Word-Prediction-LSTM
```

### 🐍 Create a Virtual Environment (Optional)

```bash
conda create -n hamlet_env python=3.10
conda activate hamlet_env
```

### 📦 Install Requirements

```bash
pip install -r requirements.txt
```

### 🚀 Run the App

```bash
streamlit run app.py
```

---

## 📚 Dataset – Hamlet

The raw Shakespearean text of *Hamlet* is used as the training corpus. It is stored in `hamlet.txt`. This dataset features:
- Rich and archaic vocabulary
- Complex syntax and structure
- Ideal for training stylistic LSTM models

---

## 🛠 Requirements

Refer to `requirements.txt` for all dependencies. Major libraries used include:

- `tensorflow==2.15.0`
- `streamlit==1.34.0`
- `numpy`, `pandas`
- `scikit-learn`, `nltk`

---

## ✨ Highlights

- Literature + AI fusion using NLP
- LSTM-based neural architecture
- Streamlit-based real-time interaction
- Shakespearean text modeling challenge

---

## 📬 Contact

For suggestions, queries, or collaboration:

- 📧 [shaileshgupta841@gmail.com]
- 🧑‍💻 [GitHub Profile](https://github.com/sg2499)

---

> Built with ❤️ using TensorFlow, Keras, and Streamlit.
