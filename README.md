# **Three-Class Sentiment Analysis (Positive • Neutral • Negative)**

This project builds a sentiment classifier that categorizes any text into **Positive**, **Neutral**, or **Negative** emotions. The model is trained on the **Twitter text_emotion dataset** and deployed using **Streamlit Cloud**, allowing users to input text and instantly view predictions along with class probabilities.

---

## **Features**

* Predicts sentiment in **three classes**

  * **Positive**
  * **Neutral**
  * **Negative**
* Interactive **Streamlit** web interface
* Displays **class probabilities**
* Preprocessing pipeline with:

  * text cleaning
  * tokenization
  * TF-IDF vectorization
* Trained using a classical ML model with saved components:

  * `sentiment_model.pkl`
  * `vectorizer.pkl`
  * `label_encoder.pkl`

---

## **Project Structure**

```
Sentiment-Analysis/
│
├── data/
│   ├── text_emotion.csv
│   ├── sentiment_dataset.csv        # cleaned + merged dataset
│   └── prepare_dataset.py           # preprocessing & label mapping
│
├── models/
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── train_model.py               # model training script
│   ├── predict.py                   # loads model + prediction function
│   └── app.py                       # Streamlit application
│
├── requirements.txt
└── README.md
```

---

## **How It Works**

### **1. Data Preparation**

The raw dataset is processed to:

* clean text
* remove noise
* reduce multiple emotion labels to **three classes**
* create a final dataset for training

### **2. Model Training**

`train_model.py` handles:

* TF-IDF vectorization
* model training
* encoding labels
* saving all components to `models/`

### **3. Prediction Pipeline**

`predict.py` loads the `model`, `vectorizer`, and `label_encoder`, then returns:

* predicted class
* probability of each class

### **4. Streamlit App**

The UI allows users to:

* type any text
* run sentiment analysis
* view results instantly

---

## **Running the App Locally**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Launch Streamlit**

```
streamlit run src/app.py
```

---

## **Deployment**

The app is deployed on **Streamlit Cloud**. 
https://sentiment-analysis-t4qjfehktslrwon4s7yzow.streamlit.app/

Files required for deployment:

* `src/app.py`
* `src/predict.py`
* `models/` (model + vectorizer + label encoder)
* `requirements.txt`

The `data/` folder is excluded from the repository for size reasons and is not required during inference.

---

## **Model Notes**

* The model is based on traditional machine learning (TF-IDF + classifier).
* Performance depends heavily on the dataset quality and label consistency.
* For stronger results, upgrading to a transformer model (e.g., DistilBERT) would be a natural next step.

---

## **Future Improvements**

* Replace TF-IDF with a transformer-based encoder
* Support multilingual sentiment analysis
* Add confidence calibration
* Expand dataset with cleaner, more balanced samples

---

## **Author**

* Rizwan Ali Mondal
