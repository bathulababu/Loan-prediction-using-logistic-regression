# Loan-prediction-using-logistic-regression

Learn how to predict if a person will be able to pay the loan with logistic regression algorithm using sklearn library for machine learning.


## Binary Classification using Logistic Regression

This project focuses on predicting loan approval using a binary classification model with Logistic Regression. The analysis includes data preprocessing, exploratory data analysis, model training, and evaluation.

### Project Description

The primary objective of this project is to develop a predictive model that can accurately determine whether a loan application will be approved or not. This is achieved by analyzing various features of the loan applicants and applying the Logistic Regression algorithm.

### Data

The dataset used in this project (`train.csv`) contains information about loan applicants. The key features include:

-   `Gender`: Gender of the applicant.
-   `Married`: Marital status of the applicant.
-   `Dependents`: Number of dependents.
-   `Education`: Education level of the applicant.
-   `Self_Employed`: Self-employment status.
-   `ApplicantIncome`: Income of the applicant.
-   `CoapplicantIncome`: Income of the co-applicant.
-   `LoanAmount`: Amount of the loan.
-   `Loan_Amount_Term`: Term of the loan.
-   `Credit_History`: Credit history of the applicant.
-   `Property_Area`: Area of the property.
-   `Loan_Status`: Whether the loan was approved or not (target variable).

### Methodology

1.  **Data Loading and Preprocessing**:
    -   The dataset is loaded using pandas.
    -   Handling missing values by imputing with the mean, median, or mode.
    -   Encoding categorical variables using one-hot encoding.
    -   Dropping irrelevant columns.
2.  **Exploratory Data Analysis (EDA)**:
    -   Analyzing the distribution of variables.
    -   Visualizing the data using plots (e.g., histograms, box plots, count plots).
    -   Understanding the relationships between different features and the target variable.
3.  **Model Training**:
    -   Splitting the dataset into training and testing sets.
    -   Scaling the numerical features using `StandardScaler`.
    -   Training a Logistic Regression model on the training data.
4.  **Model Evaluation**:
    -   Making predictions on the test set.
    -   Evaluating the model's performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
    -   Displaying the classification report.

### Libraries Used

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scikit-learn (LogisticRegression, train_test_split, StandardScaler, LabelEncoder, OneHotEncoder, SimpleImputer, accuracy_score, classification_report, confusion_matrix)
-   scipy
-   math

### Usage

To replicate or extend this project:

1.  Ensure you have the required libraries installed. You can install them using pip:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```

2.  Place the `train.csv` file in the same directory as the notebook.
3.  Run the Jupyter Notebook (`loan-prediction-project-detailed.ipynb`) to execute the code and reproduce the results.

### Files

-   `train.csv`: The dataset used for loan prediction.
-   `loan-prediction-project-detailed.ipynb`: Jupyter Notebook containing the code and analysis.

### Results

The Logistic Regression model's performance is evaluated, and the key metrics are reported in the notebook. The results provide insights into the model's ability to predict loan approval based on the given features.

### Author

\[Naveen Babu Bathula]

