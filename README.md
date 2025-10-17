# Customer Churn Prediction using Machine Learning

## 📘 Overview
This project aims to predict **customer churn** using supervised machine learning algorithms.  
It follows a standard data mining workflow — from data preprocessing and exploratory analysis to model training and evaluation.

---

## 📂 Dataset
We use the **Telco Customer Churn** dataset available on Kaggle:

- [Telco Customer Churn – Blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- [Telco Customer Churn – IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

---

## 🤖 Algorithms Used
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

---

## 🛠️ Tools & Libraries
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

## 🚀 Project Workflow
1. **Data Loading & Cleaning**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Engineering & Encoding**  
4. **Model Building & Evaluation**  
5. **Visualization of Results**

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone <your_repo_url>
cd Customer_Churn_Prediction_Project
```

### 2️⃣ Create and activate a virtual environment

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```

Then open:
```
notebooks/Customer_Churn_Prediction.ipynb
```
and run all cells to train and evaluate the models.

### 5️⃣ (Optional) Deactivate the environment
```bash
deactivate
```

---

## 📊 Expected Outputs
- Model accuracy and classification reports  
- Confusion matrix visualizations saved in the `outputs/` folder  
- Comparative performance of Logistic Regression, Decision Tree, Random Forest, and SVM  

---

## 🧱 Project Structure
```
Customer_Churn_Prediction_Project/
│
├── data/                      # Dataset files
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── outputs/                   # Generated plots & results
├── scripts/
│   ├── data_preprocessing.py
│   └── model_training.py
├── requirements.txt
└── README.md
```

---

## 👩‍💻 Team Members
- **Monika Tyagi** — monikatyagi@iisc.ac.in  
- **Sourajit Bhar** — sourajitbhar@iisc.ac.in  
