Customer Segmentation using KNN and SVM

Overview:-
This project implements customer segmentation using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) classifiers. It uses Streamlit for an interactive UI and allows users to upload two CSV files containing e-commerce customer data. The app merges the datasets, preprocesses the data, and applies machine learning models to classify customer loyalty status.

Features:-
Data Upload: Users can upload two CSV files containing customer data.
Data Preprocessing: Merging datasets, handling missing values, encoding categorical variables, and standardizing numerical features.
Clustering with K-Means: Assigns customers to clusters before classification.

Classification Models:
KNN: Optimized using GridSearchCV.
SVM: Optimized using GridSearchCV.

Performance Metrics:
Accuracy, Precision, Recall, and Confusion Matrix.

Visualizations:
Confusion matrices for both models.
Model accuracy comparison using bar charts.

Why SVM?: Explanation of why SVM might perform better than KNN in certain scenarios.

Installation:
To run this project, install the necessary dependencies using:
 - pip install -r requirements.txt
   
Running the Application:
 - Run the Streamlit app using:
   - streamlit run app.py

File Structure:

 ├── app.py               # Main Streamlit application
 ├── requirements.txt     # Required Python libraries
 ├── README.md            # Project Documentation

Dependencies:

 -Python 3.x
 -Streamlit
 -Pandas
 -NumPy
 -Matplotlib
 -Seaborn
 -Scikit-learn

How it Works:
 - Upload two CSV files containing e-commerce data.
 - Click the Submit button to process the data.
 - The app will preprocess and encode the data.
 - K-Means clustering will be applied.
 - KNN and SVM models will be trained and evaluated.
 - Results will be displayed with performance metrics and visualizations.

Expected Input Format:
 The input CSV files should have columns such as:
  - gender, income, education, region, loyalty_status, purchase_frequency, Country
  - product_category (common key for merging both datasets)
  - purchase_amount, CustomerID, InvoiceNo, StockCode, etc.

Future Improvements:
 - Allow users to choose the number of clusters dynamically.
 - Provide additional feature selection options.
 - Implement more ML models for comparison.

License:
 - This project is open-source and available under the MIT License.

