# Diabetes Prediction from Medical Records

<p align="center">
  <img src="https://github.com/Nandaniipriya/Disease-Prediction-Using-Machine-Learning/raw/main/assests/banner.png" alt="Banner Image" width="80%">
</p>

## Project Overview
Diabetes mellitus is a group of chronic metabolic disorders characterized by elevated blood glucose levels, which can lead to serious health complications if left unmanaged. Early and accurate prediction of diabetes is crucial for healthcare providers to implement timely interventions and improve patient outcomes.

This project aims to develop a robust machine learning model that can reliably predict the presence of diabetes based on patient medical records. By leveraging advanced data analysis and modeling techniques, we can gain valuable insights into the key factors that contribute to the development of this disease, which can inform preventive strategies and guide healthcare decision-making.

The dataset used in this project is the Pima Indians Diabetes Database, a well-known and widely used dataset for diabetes prediction tasks. This dataset contains various medical measurements for a population of Pima Indian women, along with the information on whether they have diabetes or not.

## Dataset Overview
The Pima Indians Diabetes Database consists of the following features:

- **Pregnancies**: Number of times the patient has been pregnant
- **Glucose**: Plasma glucose concentration (from an oral glucose tolerance test)
- **BloodPressure**: Diastolic blood pressure (in mm Hg)
- **SkinThickness**: Triceps skin fold thickness (in mm)
- **Insulin**: 2-Hour serum insulin (in mu U/ml)
- **BMI**: Body mass index (weight in kg divided by height in meters squared)
- **DiabetesPedigreeFunction**: A function that scores the likelihood of diabetes based on family history
- **Age**: Age of the patient (in years)
- **Outcome**: Class variable indicating whether the patient has diabetes (1) or not (0)

This dataset provides a comprehensive set of medical measurements that can potentially be used to predict the presence of diabetes. However, the dataset also contains some missing values, which need to be handled appropriately during the data preprocessing stage.

## Methodology
The project follows a structured data analysis and machine learning pipeline, consisting of the following key steps:

1. **Data Inspection and Cleaning**:
   - Thoroughly inspect the dataset to identify and understand any missing values or anomalies.
   - Implement appropriate data imputation techniques to handle the missing data, ensuring that the original data distribution is preserved.
   - Perform additional data cleaning and transformation as needed, such as handling outliers or converting feature types.

2. **Exploratory Data Analysis**:
   - Conduct a detailed exploratory analysis to understand the relationships between the features and the target variable (Outcome).
   - Generate visualizations, such as histograms, scatter plots, and correlation matrices, to gain insights into the data.
   - Identify any potential feature engineering opportunities that could improve the model's performance.

3. **Feature Engineering**:
   - Based on the insights from the exploratory data analysis, create new features or transform existing ones to enhance the predictive power of the model.
   - Consider feature selection techniques, such as correlation analysis or recursive feature elimination, to identify the most important predictors of diabetes.

4. **Model Selection and Evaluation**:
   - Train and evaluate a variety of classification models, including Logistic Regression, Decision Trees, Random Forests, and XGBoost.
   - Assess the performance of the models using appropriate metrics, such as accuracy, precision, recall, and F1-score, to determine the best-performing model.
   - Employ techniques like cross-validation to ensure the model's robustness and generalization capabilities.

5. **Feature Importance Analysis**:
   - Examine the feature importances of the best-performing model to understand which medical measurements are most influential in predicting the presence of diabetes.
   - Interpret the feature importance results in the context of the existing medical knowledge about diabetes risk factors.

6. **Model Deployment and Evaluation**:
   - Finalize the best-performing model and package it for easy deployment in a real-world setting.
   - Evaluate the model's performance on new, unseen data to ensure its reliability and effectiveness in practical applications.
   - Document the model's limitations and potential areas for future improvement.

<p align="center">
  <img src="https://github.com/Nandaniipriya/Disease-Prediction-Using-Machine-Learning/raw/main/assests/workflow.png" alt="Workflow" width="80%">
</p>

## Results and Findings
The best-performing model in our experiments was the XGBoost Classifier, which achieved an accuracy of approximately 82% on the test set. The most important features for predicting diabetes were:

1. **Body Mass Index (BMI)**
2. **Diabetes Pedigree Function**
3. **Glucose**
4. **Age**

<p align="center">
  <img src="https://github.com/Nandaniipriya/Disease-Prediction-Using-Machine-Learning/raw/main/assests/confusion-matrix.png" alt="Confusion-matrix" width="80%">
</p>

<p align="center">
  <img src="https://github.com/Nandaniipriya/Disease-Prediction-Using-Machine-Learning/raw/main/assests/decision-tree.png" alt="Decision-tree" width="80%">
</p>

The feature importance analysis reveals that factors like body mass index, genetic predisposition, and blood glucose levels are the key predictors of diabetes, aligning with our understanding of the disease. These insights can be valuable for healthcare professionals in identifying high-risk individuals and implementing targeted preventive strategies.

Furthermore, the high accuracy achieved by the XGBoost Classifier demonstrates the power of advanced machine learning techniques in predicting the presence of diabetes. By leveraging the predictive capabilities of this model, healthcare providers can potentially improve early detection, leading to timely interventions and better patient outcomes.

## Usage and Installation
To run the code for this project, you will need to have the following dependencies installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

You can install these dependencies using pip:
```
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

Once the dependencies are installed, you can run the Jupyter Notebook file to see the complete analysis and results.

## Conclusion and Future Work
This project has successfully demonstrated the potential of machine learning in predicting the presence of diabetes based on medical records. The insights gained from the feature importance analysis can be valuable for healthcare professionals in understanding the key factors that contribute to the development of this chronic disease.

By leveraging advanced machine learning techniques, such as the XGBoost Classifier, we were able to achieve a high level of accuracy in predicting diabetes, highlighting the promise of data-driven approaches in the field of healthcare. The findings from this project can serve as a starting point for further research and development in the area of diabetes prediction and prevention.

In the future, we plan to expand this project in the following ways:

1. **Explore Additional Datasets**: Investigate the performance of the model on other diabetes-related datasets, potentially from different demographics or healthcare settings, to assess its generalization capabilities.

2. **Incorporate Additional Features**: Explore the inclusion of other relevant medical measurements or lifestyle factors that could further improve the model's predictive power.

3. **Develop a Production-Ready Application**: Package the best-performing model into a user-friendly application that can be easily deployed in healthcare organizations, enabling real-world implementation and continuous evaluation.

4. **Collaborate with Domain Experts**: Engage with healthcare professionals and domain experts to gather feedback, refine the model, and ensure that the project's findings align with clinical best practices and guidelines.

By continuously improving and expanding this project, we aim to contribute to the ongoing efforts in the field of diabetes prediction and prevention, ultimately supporting healthcare providers in delivering better patient care and improving population health outcomes.
