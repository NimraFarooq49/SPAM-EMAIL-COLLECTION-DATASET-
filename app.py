import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# 🌟 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="ML Analysis Platform",
    page_icon="🤖",
    layout="wide"
)

# ==========================================================
# 🎨 CUSTOM CSS
# ==========================================================
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .section-header {
        color: #2ecc71;
        font-size: 2em;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# 📌 TITLE
# ==========================================================
st.markdown('<p class="main-header">🤖 Machine Learning Analysis Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Complete EDA, Visualization & Model Training Solution</p>', unsafe_allow_html=True)
st.markdown("---")

# ==========================================================
# 📁 FILE UPLOAD SECTION
# ==========================================================
st.markdown('<p class="section-header">📁 Step 1: Upload Dataset</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ Dataset loaded successfully!")
    
    # Show raw data preview
    with st.expander("👀 Preview Dataset (First 10 Rows)"):
        st.dataframe(df.head(10))
    
    # ==========================================================
    # 📊 EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📊 Step 2: Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Basic Information
    with st.expander("📋 Dataset Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Names:**")
            st.write(list(df.columns))
            
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes)
    
    # Summary Statistics
    with st.expander("📈 Summary Statistics"):
        st.write(df.describe())
    
    # Missing Values Analysis
    with st.expander("🔍 Missing Values Analysis"):
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            st.write("**Columns with Missing Values:**")
            st.write(missing_data)
            
            # Visualize missing values
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_data.plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Missing Values by Column')
            ax.set_ylabel('Count')
            ax.set_xlabel('Columns')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.success("✅ No missing values found in the dataset!")
    
    # ==========================================================
    # 📉 VISUALIZATIONS
    # ==========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">📉 Step 3: Data Visualizations</p>', unsafe_allow_html=True)
    
    # Select numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Histograms
        with st.expander("📊 Histograms (Distribution of Numeric Features)"):
            selected_hist_cols = st.multiselect(
                "Select columns for histograms:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_hist_cols:
                n_cols = min(2, len(selected_hist_cols))
                n_rows = (len(selected_hist_cols) + 1) // 2
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten() if len(selected_hist_cols) > 1 else [axes]
                
                for idx, col in enumerate(selected_hist_cols):
                    axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
                
                # Hide extra subplots
                for idx in range(len(selected_hist_cols), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Box Plots
        with st.expander("📦 Box Plots (Outlier Detection)"):
            selected_box_cols = st.multiselect(
                "Select columns for box plots:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                key='box'
            )
            
            if selected_box_cols:
                n_cols = min(2, len(selected_box_cols))
                n_rows = (len(selected_box_cols) + 1) // 2
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = axes.flatten() if len(selected_box_cols) > 1 else [axes]
                
                for idx, col in enumerate(selected_box_cols):
                    axes[idx].boxplot(df[col].dropna(), vert=True)
                    axes[idx].set_title(f'Box Plot of {col}')
                    axes[idx].set_ylabel(col)
                
                # Hide extra subplots
                for idx in range(len(selected_box_cols), len(axes)):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Correlation Heatmap
        if len(numeric_cols) > 1:
            with st.expander("🔥 Correlation Heatmap"):
                fig, ax = plt.subplots(figsize=(12, 8))
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, linewidths=1)
                ax.set_title('Correlation Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
    
    # Count Plots for Categorical Features
    if len(categorical_cols) > 0:
        with st.expander("📊 Count Plots (Categorical Features)"):
            selected_cat_col = st.selectbox(
                "Select a categorical column:",
                categorical_cols
            )
            
            if selected_cat_col:
                fig, ax = plt.subplots(figsize=(12, 6))
                value_counts = df[selected_cat_col].value_counts()
                
                # Limit to top 20 categories if too many
                if len(value_counts) > 20:
                    st.warning("Showing top 20 categories only")
                    value_counts = value_counts.head(20)
                
                value_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
                ax.set_title(f'Count Plot of {selected_cat_col}')
                ax.set_xlabel(selected_cat_col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
    
    # ==========================================================
    # 🤖 MODEL TRAINING SECTION
    # ==========================================================
    st.markdown("---")
    st.markdown('<p class="section-header">🤖 Step 4: Model Training & Evaluation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column
        target_column = st.selectbox(
            "Select Target Column (y):",
            df.columns.tolist()
        )
    
    with col2:
        # Select model
        model_choice = st.selectbox(
            "Select Machine Learning Model:",
            ["Logistic Regression", "Support Vector Machine (SVM)", 
             "Random Forest", "K-Nearest Neighbors (KNN)"]
        )
    
    # Feature selection
    feature_columns = st.multiselect(
        "Select Feature Columns (X):",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column][:min(5, len(df.columns)-1)]
    )
    
    # Test size slider
    test_size = st.slider("Test Set Size (%):", 10, 50, 20) / 100
    
    # Train button
    if st.button("🚀 Train Model", type="primary"):
        if not feature_columns:
            st.error("⚠️ Please select at least one feature column!")
        else:
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
                    
                    # Encode categorical variables
                    le_dict = {}
                    for col in X.select_dtypes(include=['object']).columns:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        le_dict[col] = le
                    
                    # Encode target if categorical
                    if y.dtype == 'object':
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y.astype(str))
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Select and train model
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif model_choice == "Support Vector Machine (SVM)":
                        model = SVC(kernel='rbf', random_state=42)
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:  # KNN
                        model = KNeighborsClassifier(n_neighbors=5)
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Handle binary vs multiclass
                    avg_method = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                    recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                    
                    # Display results
                    st.success("✅ Model trained successfully!")
                    
                    st.markdown("### 📊 Model Performance Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1-Score", f"{f1:.2%}")
                    
                    # Confusion Matrix
                    st.markdown("### 🎯 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
                    
                    # Classification Report
                    with st.expander("📋 Detailed Classification Report"):
                        st.text(classification_report(y_test, y_pred))
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
                    st.info("Please check your data and feature selection.")

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    st.markdown("### 📝 Instructions:")
    st.markdown("""
    1. Upload your CSV dataset
    2. Explore the data through EDA section
    3. View various visualizations
    4. Select target and feature columns
    5. Choose a machine learning model
    6. Train and evaluate the model
    """)

# ==========================================================
# 📌 FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:#666;'>
    Made with ❤️ using Streamlit & Scikit-learn | 
    <a href='https://streamlit.io' target='_blank'>Streamlit</a> | 
    <a href='https://scikit-learn.org' target='_blank'>Scikit-learn</a>
    </p>
""", unsafe_allow_html=True)
