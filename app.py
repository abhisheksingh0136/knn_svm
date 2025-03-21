import streamlit as st # Importing Streamlit for creating a web application
import pandas as pd # Importing pandas for data manipulation
import matplotlib.pyplot as plt # Importing Matplotlib for visualization
import seaborn as sns # Importing Seaborn for enhanced visualization
from sklearn.model_selection import train_test_split, GridSearchCV # Importing train-test split and GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler # Importing LabelEncoder and StandardScaler for data preprocessing
from sklearn.neighbors import KNeighborsClassifier # Importing KNN Classifier
from sklearn.svm import SVC # Importing Support Vector Classifier (SVC)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score # Importing performance metrics
from sklearn.cluster import KMeans # Importing KMeans for clustering

# Streamlit app configuration
st.set_page_config(page_title="🎯Customer Segmentation with KNN & SVM", layout="wide")
st.title("🎯Customer Segmentation using KNN and SVM")

# File uploader widgets to upload two CSV files
uploaded_file1 = st.file_uploader("Upload First CSV File", type=["csv"])
uploaded_file2 = st.file_uploader("Upload Second CSV File", type=["csv"])

def process_data(file1, file2):
    df1 = pd.read_csv(file1) # Reading first CSV file
    df2 = pd.read_csv(file2) # Reading second CSV file
    merged_df = pd.merge(df1, df2, on="product_category", how="inner") # Merging datasets on 'product_category'
    merged_df["purchase_amount"].fillna(merged_df["purchase_amount"].median(), inplace=True) # Handling missing values
    
    # Encoding categorical columns
    categorical_columns = ["gender", "income", "education", "region", "loyalty_status", "purchase_frequency", "Country"]
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        merged_df[col] = le.fit_transform(merged_df[col]) # Applying label encoding
        label_encoders[col] = le # Storing encoders
        
    # Dropping unnecessary columns
    drop_columns = ["id", "InvoiceNo", "StockCode", "Description", "InvoiceDate", "CustomerID", "product_category"]
    merged_df = merged_df.drop(columns=drop_columns)
    return merged_df # Returning preprocessed dataframe

# Processing the files after upload
if uploaded_file1 and uploaded_file2:
    
        if st.button("Submit"): # Button to trigger processing
            df = process_data(uploaded_file1, uploaded_file2) # Process data
            X = df.drop(columns=["loyalty_status"]) # Features (independent variables)
            y = df["loyalty_status"]  # Target variable (dependent variable)
            
            # K-Means clustering for segmentation
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans_labels = kmeans.fit_predict(X)
            X["Cluster"] = kmeans_labels  # Adding cluster labels to dataset
            
            # Splitting data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Standardizing the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # KNN model tuning using GridSearchCV
            knn_params = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
            grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5)
            grid_knn.fit(X_train_scaled, y_train)
            best_knn = grid_knn.best_estimator_  # Selecting best KNN model
            
            # KNN predictions
            y_pred_knn = best_knn.predict(X_test_scaled)
            knn_accuracy = accuracy_score(y_test, y_pred_knn) # Accuracy score
            knn_conf_matrix = confusion_matrix(y_test, y_pred_knn) # Confusion matrix
            knn_precision = precision_score(y_test, y_pred_knn, average='weighted') # Precision score
            knn_recall = recall_score(y_test, y_pred_knn, average='weighted') # Recall score
            knn_report = classification_report(y_test, y_pred_knn) # Classification report
            
            # SVM model tuning using GridSearchCV
            svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
            grid_svm = GridSearchCV(SVC(), svm_params, cv=5)
            grid_svm.fit(X_train_scaled, y_train)
            best_svm = grid_svm.best_estimator_  # Selecting best SVM model
            
            # SVM predictions
            y_pred_svm = best_svm.predict(X_test_scaled)
            svm_accuracy = accuracy_score(y_test, y_pred_svm) # Accuracy score
            svm_conf_matrix = confusion_matrix(y_test, y_pred_svm) # Confusion matrix
            svm_precision = precision_score(y_test, y_pred_svm, average='weighted') # Precision score
            svm_recall = recall_score(y_test, y_pred_svm, average='weighted') # Recall score
            svm_report = classification_report(y_test, y_pred_svm) # Classification report
            
            # Displaying results in Streamlit
            st.subheader("KNN Results")
            st.write(f"**Best Parameters:** {grid_knn.best_params_}")
            st.write(f"**Accuracy:** {knn_accuracy:.4f}")
            st.write(f"**Precision:** {knn_precision:.4f}")
            st.write(f"**Recall:** {knn_recall:.4f}")
            st.subheader("Classification Report:")
            st.text(knn_report)
            st.subheader("SVM Results")
            st.write(f"**Best Parameters:** {grid_svm.best_params_}")
            st.write(f"**Accuracy:** {svm_accuracy:.4f}")
            st.write(f"**Precision:** {svm_precision:.4f}")
            st.write(f"**Recall:** {svm_recall:.4f}")
            st.subheader("Classification Report:")
            st.text(svm_report)
            
            # Model comparison
            st.subheader(" 📊 Model Comparison")
            st.write(f"KNN Accuracy: {knn_accuracy:.4f}, SVM Accuracy: {svm_accuracy:.4f}")
            st.write(f"KNN Precision: {knn_precision:.4f}, SVM Precision: {svm_precision:.4f}")
            st.write(f"KNN Recall: {knn_recall:.4f}, SVM Recall: {svm_recall:.4f}")
            
            # Visualization
            st.subheader("📊 Visualizations")
            fig, ax = plt.subplots()
            sns.heatmap(knn_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("KNN Confusion Matrix")
            st.pyplot(fig)
            fig, ax = plt.subplots()
            sns.heatmap(svm_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title("SVM Confusion Matrix")
            st.pyplot(fig)
            models = ['KNN', 'SVM']
            accuracy_scores = [knn_accuracy, svm_accuracy]
            fig, ax = plt.subplots()
            sns.barplot(x=models, y=accuracy_scores, palette='coolwarm', ax=ax)
            ax.set_title("Accuracy Comparison")
            st.pyplot(fig)
            st.write(" ")
            # Explanation of SVM's performance
            st.subheader("🧐 Why SVM has Higher Accuracy?")
            
            st.markdown("""
                - **Optimal Hyperplane & Margin Maximization**: SVM finds the best hyperplane with the largest margin, improving generalization.
                - **Handles High-Dimensional Data**: Works better with many features, while KNN struggles with distance calculations.
                - **Non-Linear Decision Boundaries**: SVM can map data to a higher-dimensional space for better separation.
                - **Less Sensitive to Noise**: SVM focuses only on key support vectors, reducing the impact of outliers.
                - **Better for Imbalanced Data**: KNN may favor the majority class, while SVM optimizes class separation.
                """)