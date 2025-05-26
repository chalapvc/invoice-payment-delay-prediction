### Invoice Payment Delay Prediction

**Author**: Venkata Chalapathy Pidathala

### Executive summary
This project focuses on developing a machine learning model to predict delays in invoice payments. By analyzing historical transaction data, the model aims to assist businesses in managing cash flow more effectively and improving accounts receivable processes.

### Rationale
Late invoice payments can disrupt a company's financial stability. This project seeks to predict which invoices are likely to be paid late, enabling proactive measures to mitigate potential cash flow issues.

### Research Question
Can we accurately predict whether an invoice will be paid on time based on customer behavior and transaction characteristics?

### Data Sources
The dataset used in this project contains detailed customer and loan-related information. It includes the following features:
  - `person_age`: Age of the customer (e.g., 24, 27, 30).
  - `person_income`: Annual income in USD (e.g., 28,000; 64,000).
  - `person_home_ownership`: Home ownership status — options include `OWN`, `RENT`, and `MORTGAGE`.
  - `person_emp_length`: Length of employment (in years), reflecting job stability (e.g., 6.0, 0.0, 10.0).
  - `loan_intent`: Reason for the loan, such as `HOMEIMPROVEMENT`, `PERSONAL`, `EDUCATION`, `DEBTCONSOLIDATION`, or `MEDICAL`.
  - `loan_grade`: Credit grade assigned to the customer (e.g., A, B, C, D, E).
  - `loan_amnt`: Loan amount requested or invoiced (in USD), e.g., 10,000; 13,000; 16,000.
  - `loan_int_rate`: Interest rate for the loan (some values may be missing).
  - `loan_status`: Loan repayment status — `0` indicates on-time payment, `1` indicates delayed payment.
  - `loan_percent_income`: Ratio of the loan amount to the customer’s income (e.g., 0.36; 0.16).
  - `cb_person_default_on_file`: Indicates if the customer has a default history — `Y` for yes, `N` for no.
  - `cb_person_cred_hist_length`: Length of the customer’s credit history (in years), e.g., 2, 10, 6.

### Data Summary
  - **Total Rows**: 6,516 customer records
  - **Total Columns**: 12 features:
    - 5 numerical (integers)
    - 3 floating-point (decimals)
    - 4 categorical (textual or class-based values)
  - **Missing Values**: Some columns contain missing data, notably `loan_int_rate`.

This dataset provides a comprehensive view of customer demographics, transaction information, and payment behaviors, which form the basis for predicting invoice delays.

### Methodology
1. Collect and preprocess historical invoice and payment data.
2. Identify relevant features such as customer payment history, invoice amount, and transaction patterns.
3. Train and evaluate machine learning models for prediction accuracy.
4. Interpret results and suggest actionable strategies based on findings.

### Project workflow
The project was executed in the following key stages:
1. **Data Preprocessing**
   - **Handling Missing Values**: Dropped columns with excessive missing data and applied imputation techniques (e.g., mean/mode) for minor gaps.
   - **Feature Engineering**: Created new features such as `invoice-to-income` ratio to enhance predictive performance.
   - **Column Standardization**: Renamed columns for improved readability and consistency.
   - **Categorical Encoding**: Converted categorical features into numeric formats using binary and one-hot encoding, preparing them for machine learning algorithms.

2. **Exploratory Data Analysis (EDA)**
    - **Distribution Analysis**: Plotted histograms and boxplots to understand distributions and detect outliers in numerical data.
    - **Categorical Variable Insights**: Used count plots to explore trends in home ownership, loan intent, and employment length.
    - **Correlation Matrix**: Generated a heatmap to identify multicollinearity and inform feature selection strategies.
  
3. **Model Training & Evaluation**
   - **Algorithms Applied:**
     - Random Forest
     - Gradient Boosting
     - Neural Network (Keras)
   - **Hyperparameter Tuning**: Used grid search and cross-validation to fine-tune model parameters for optimal performance.
   - **Evaluation Metrics**: Assessed models using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix

4. **Model Selection**
   - **Feature Importance**: Evaluated feature contribution using model-based importance metrics.
   - **Model Comparison**: Compared performance across all models and selected the best-performing one for deployment or recommendation.

### Project Code Workflow

1. **Importing Required Libraries**
   ```bash
   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt

   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.preprocessing import StandardScaler, LabelEncoder, SimpleImputer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, confusion_matrix

   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense, Dropout
   from tensorflow.keras.callbacks import EarlyStopping

   import tensorflow as tf
   
   ```
   **Purpose of Each Library/Function:**
     - `pandas`, `numpy`: For data manipulation, analysis, and handling arrays.
     - `seaborn`, `matplotlib.pyplot`: For visualization — to explore distributions, patterns, and correlations.
     - `train_test_split`, `GridSearchCV`: For splitting data into train/test sets and tuning hyperparameters efficiently.
     - `StandardScaler`, `LabelEncoder`, `SimpleImputer`: For preprocessing — scaling, encoding categorical values, and handling missing data.
     - `RandomForestClassifier`: A robust tree-based ensemble algorithm for classification tasks.
     - `classification_report`, `confusion_matrix`: To assess model performance.
     - `Sequential`, `Dense`, `Dropout`: Layers used to construct a feedforward neural network.
     - `EarlyStopping`: A Keras callback to halt training when validation loss no longer improves, reducing overfitting.
     - `tensorflow` **(tf)**: The core framework used to build and train the deep learning model.

2. **Load and Explore the Dataset**
   ```bash
   # Load the dataset
   df = pd.read_csv('data/credit_risk_dataset.csv')

   # Basic exploration
   df.head()             # Preview first few rows
   df.info()             # Data types and non-null counts
   df.describe()         # Summary statistics
   df.isnull().sum()     # Count of missing values per column
   ```
   **Key Steps:**
     - `pd.read_csv()`: Loads the dataset into a DataFrame.
     - `df.head()`: Displays the top rows to get a sense of the data structure.
     - `df.info()`: Identifies data types and missing values.
     - `df.describe()`: Reveals distributions and potential outliers in numerical columns.
     - `df.isnull().sum()`: Pinpoints missing values that need imputation.

3. **Data Visualization**
    ```bash
    # Plot distributions for numeric features
    num_cols = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
    
    # Plot target variable
    sns.countplot(x='loan_status', data=df)
    plt.title('Distribution of Loan Status')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.show()
    ```
    **What’s Happening:**
     - **Distribution Plots**: Reveal skewness, outliers, or normality of numeric features.
     - **Target Count Plot**: Shows class balance between on-time and delayed payments.
     - **Heatmap**: Helps detect multicollinearity and relationships between numeric features.

4. **Data Preprocessing**
   - Missing Value Handling
   - Dropped columns with more than **30% missing values**.
   - Applied imputation on remaining columns:
     - **Numerical columns** → filled with **mean**.
     - **Categorical columns** → filled with **most frequent** value.
   - **Encoding Categorical Variables**
     - **Binary Encoding**: Applied `LabelEncoder` to binary columns like `historical_default`.
     - **One-Hot Encoding**: Used `pd.get_dummies()` for multi-class features such as:
       - `home_ownership`
       - `purchase_intent`
       - `credit_grade`
     - **Feature Scaling**
       - Standardized all numerical features using `StandardScaler`.
       - This ensures the data is normalized, which is especially beneficial for neural network performance.

5. **Feature Engineering**
   - **Column Renaming**
     - Renamed columns to follow consistent and meaningful naming conventions for better readability (e.g., `cb_person_cred_hist_length → credit_history_years`).
   - **New Feature Creation**
     - Created a new derived feature:
        ```bash
        df['invoice_to_income_ratio'] = df['loan_amnt'] / df['person_income']
        ```
     - This ratio helps assess **financial risk** relative to income.

6. **Train-Test Split**
   - **Define Inputs and Target**
     - `X` → Feature matrix (independent variables)
     - `y` → Target variable (`loan_status`: 0 = on-time, 1 = delayed)
   - Split the Dataset
     - Used `train_test_split()` with an **80/20 ratio**:
       ```bash
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       ```
7. **Random Forest with Hyperparameter Tuning**
   - **Define Parameter Grid**
     - Specified combinations for model tuning:
       ```bash
       param_grid = {
         'n_estimators': [100, 200],
         'max_depth': [5, 10, None],
       'min_samples_split': [2, 5]
       }
       ```
   - **Apply GridSearchCV**
     - Performed **5-fold cross-validation**:
       ```bash
       grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
       grid_search.fit(X_train, y_train)
       ```
     - **Evaluate the Model**
       - Displayed:
         - Best parameters
         - Cross-validation score
         - Classification report and confusion matrix for the test set
           ```bash
           print("Best Parameters:", grid_search.best_params_)
           print("CV Accuracy:", grid_search.best_score_)

           y_pred = grid_search.predict(X_test)
           print(classification_report(y_test, y_pred))
           sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
           ```
8. **Feature Importance Visualization**
    - **Visualizing Feature Importance**
      - Used `best_model.feature_importances_` from the trained **Random Forest model**.
      - Plotted a bar chart to highlight the top features contributing to the predictions.
      - This step helps interpret which variables are most influential in determining **loan payment delays**.

9. **Neural Network Model**
    - **Model Architecture (Sequential API)**
      - Built using `tensorflow.keras.Sequential`.
      - **Input Layer**: 64 neurons, ReLU activation.
      - **Hidden Layers**: Two additional dense layers with ReLU activation and `Dropout()` for regularization.
      - **Output Layer**: Single neuron with **sigmoid activation** for binary classification (`loan_status`).
    - **Compilation Settings**
      - **Optimizer**: Adam
      - **Loss Function**: Binary Crossentropy
      - **Metrics**: Accuracy
    - **Early Stopping**
      - Implemented `EarlyStopping` to monitor validation loss.
      - Prevents overfitting by halting training when performance no longer improves.

10 **Train the Neural Network**
   - **Training Settings**
     - Trained on `X_train` and `y_train` for 30 epochs.
     - Used a **validation split** (e.g., 0.2) and a **batch size** of 32.
     - The training process returns a `history` object containing accuracy and loss values per epoch.
   - **Monitoring Progress**
     - Stored training metrics in `history.history`.
     - Used `matplotlib` to plot:
       - Training vs. validation **accuracy**
       - Training vs. validation **loss**
 
 11 **Neural Network Evaluation**
   - **Model Evaluation on Test Data**
     - Evaluated performance on the unseen test set (`X_test`, `y_test`).
     - Reported final **test accuracy**.
   - **Training History Visualization**
     - Plotted training curves to assess:
       - **Model convergence**
       - Signs of **overfitting** or **underfitting**

### Results
The models demonstrated strong performance on the dataset, with the neural network achieving high predictive accuracy and maintaining generalization across training and validation sets.

- **Neural Network Model Performance**
  - **Accuracy**
    - Achieved **over 90% accuracy** on both training and validation sets.
    - Indicates that the model has effectively **learned the underlying patterns** without significant overfitting.
  - **Loss Trend**
    - **Training and validation loss** decreased consistently over 30 epochs.
    - Suggests good **model convergence** and **generalization capability**.

  ![result_data](/visuals/Neural_Network_Model_Results.png)

- **Model Evaluation Results**
  - Random Forest Classifier
  - Best Hyperparameters
    - `max_depth=20`
    - `min_samples_leaf=1`
    - `min_samples_split=2`
    - `n_estimators=100`
  - **Cross-Validation Score: 0.7028**
  - **Confusion Matrix**
    |   | Predicted: 0 | Predicted: 1 |
    | ------------- | ------------- | ------------- |
    | **Actual: 0**  | 984 | 19   |
    | **Actual: 1**  | 98  | 203  |

  - **Classification Report**
    
    | Metric  | Class 0 | Class 1 |
    | ------------- | ------------- | ------------- |
    | **Precision**  | 91%  | 91%  |
    | **Recall**  | 98%  | 67%  |
    | **F1-Score**  | 94%  | 78%  |

  - **Overall Accuracy: 91%**

- **Gradient Boosting Classifier**
  - Best Hyperparameters
    - `learning_rate=0.1`
    - `max_depth=5`
    - `n_estimators=100`
  - **Cross-Validation Score: 0.7116**
  - **Confusion Matrix**
    |   | Predicted: 0 | Predicted: 1 |
    | ------------- | ------------- | ------------- |
    | **Actual: 0** | 984 | 19   |
    | **Actual: 1**  | 92  | 209  |

  - **Classification Report**
    
    | Metric  | Class 0 | Class 1 |
    | ------------- | ------------- | ------------- |
    | Precision  | 91%  | 92%  |
    | Recall  | 98%  | 69%  |
    | F1-Score  | 95%  | 79%  |
    
  - **Overall Accuracy: 91%**

  ![result_data](/visuals/Gradient_Boosting_Model_Results.png)

### Summary
Both the **Random Forest** and **Gradient Boosting** models achieved an overall accuracy of **91%**, with **Gradient Boosting** showing slightly better performance in terms of recall and F1-score for delayed payments. The **Neural Network** also demonstrated high accuracy, with a smooth training curve indicating strong generalization and minimal overfitting.

These results suggest that the models are well-suited for predicting **invoice payment delays**, offering reliable performance across both timely and delayed payments.

### Programming & Libraries
**Installation and Usage**
  - **Installation:**
    ```bash
      git clone https://github.com/chalapvc/invoice-payment-delay-prediction.git
      cd invoice-payment-delay-prediction
      pip install -r requirements.txt
    
       ```
  - **Usage:**
    Open `invoice_payment_delay.ipynb` in Jupyter Notebook and follow the steps to preprocess data, train models, and evaluate results.

**Python**
  - `pandas`, `numpy` – Data manipulation and numerical operations
  - `scikit-learn` – Machine learning models, preprocessing, evaluation
  - `seaborn`, `matplotlib` – Data visualization
  - `tensorflow.keras` – Neural network architecture and training

**Machine Learning Models**
  - **Random Forest Classifier**
  - **Gradient Boosting Classifier**
  - **Neural Network (Keras Sequential API)**

**Environment**
  - **Jupyter Notebook**
    - For data exploration, visualization, and model development in an interactive format

### Next steps
To enhance the project further, here are some suggested next steps:
1. **Model Optimization**
   - Fine-tune the neural network architecture using advanced techniques like dropout schedules, batch normalization, or learning rate schedulers.
2. **Feature Expansion**
   - Include external datasets (e.g., credit scores, payment history over time) to enrich the feature space and improve prediction accuracy.
3. **Class Imbalance Handling**
   - Apply techniques like SMOTE (Synthetic Minority Over-sampling Technique) or class weighting to improve recall for minority classes (e.g., delayed payments).
4. **Model Explainability**
   - Use SHAP or LIME to provide interpretability for model predictions, helping stakeholders understand feature importance on an individual prediction level.
5. **Automated Pipeline**
   - Convert the workflow into a reproducible pipeline using libraries like MLflow, DVC, or sklearn pipelines for version control and scalability.

### Outline of project
This project is organized into multiple notebooks, each addressing a specific phase of the workflow:

**Notebook 1 – Data Exploration & Preprocessing**
Covers data loading, missing value treatment, feature engineering, encoding, and data scaling.

**Notebook 2 – Model Building & Evaluation**
Implements machine learning models (Random Forest, Gradient Boosting, Neural Network), hyperparameter tuning, and performance evaluation.

**Notebook 3 – Results & Visualization**
Includes visualizations for feature importance, confusion matrices, training curves, and summary metrics across models.

### References
1. [Predicting Late Payments Using Decision Tree-Based Algorithms](https://arthurflor23.medium.com/prediction-of-late-payments-using-decision-tree-based-algorithms-ce72a2fbccab)
2. [Optimize Cash Collection: Use Machine Learning to Predict Invoice Payment](https://arxiv.org/abs/1912.10828)

### Contact and Further Information
If you have any questions, feedback, or would like to collaborate on similar projects, feel free to get in touch:

**Name**: Venkata Chalapathy Pidathala<br/>
**Email**: chalapvc@gmail.com<br/>
**LinkedIn**: https://www.linkedin.com/in/venkata-chalapathy-pidathala-4232019/<br/>
**GitHub**: github.com/chalapvc<br/>

This project was developed as part of an academic or self-learning initiative, showcasing end-to-end data science and machine learning techniques. Contributions, suggestions, and forks are always welcome!

And data for this project was sourced from [Kaggle's Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
