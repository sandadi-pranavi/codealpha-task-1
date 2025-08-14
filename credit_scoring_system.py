import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic credit data for demonstration
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'debt_to_income_ratio': np.random.uniform(0, 1, n_samples),
        'credit_history_length': np.random.randint(0, 30, n_samples),
        'num_credit_accounts': np.random.randint(1, 15, n_samples),
        'payment_history_score': np.random.uniform(300, 850, n_samples),
        'credit_utilization': np.random.uniform(0, 1, n_samples),
        'num_late_payments': np.random.poisson(2, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.normal(25000, 15000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure positive values where needed
    df['income'] = np.abs(df['income'])
    df['loan_amount'] = np.abs(df['loan_amount'])
    
    # Create target variable based on logical rules
    df['credit_score'] = (
        0.3 * (df['payment_history_score'] / 850) +
        0.2 * (1 - df['debt_to_income_ratio']) +
        0.15 * (df['credit_history_length'] / 30) +
        0.15 * (1 - df['credit_utilization']) +
        0.1 * (df['income'] / 100000) +
        0.1 * (1 - df['num_late_payments'] / 10)
    )
    
    # Binary classification: 1 = Good Credit, 0 = Bad Credit
    df['creditworthy'] = (df['credit_score'] > 0.6).astype(int)
    
    return df

# Generate sample data
df = generate_sample_data(1000)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


def perform_eda(df):
    """
    Perform exploratory data analysis
    """
    print("=== EXPLORATORY DATA ANALYSIS ===")
    
    # Basic statistics
    print("\nDataset Info:")
    print(df.info())
    
    print("\nTarget Variable Distribution:")
    print(df['creditworthy'].value_counts())
    print(f"Creditworthy percentage: {df['creditworthy'].mean():.2%}")
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return df

# Perform EDA
df_analyzed = perform_eda(df)


def feature_engineering(df):
    """
    Create new features from existing data
    """
    print("=== FEATURE ENGINEERING ===")
    
    df_fe = df.copy()
    
    # Create new features
    df_fe['income_to_loan_ratio'] = df_fe['income'] / (df_fe['loan_amount'] + 1)
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=[0, 25, 35, 50, 100], 
                               labels=['Young', 'Adult', 'Middle', 'Senior'])
    df_fe['income_category'] = pd.cut(df_fe['income'], bins=3, 
                                     labels=['Low', 'Medium', 'High'])
    df_fe['debt_burden'] = df_fe['debt_to_income_ratio'] * df_fe['credit_utilization']
    df_fe['credit_experience'] = df_fe['credit_history_length'] * df_fe['num_credit_accounts']
    df_fe['payment_reliability'] = df_fe['payment_history_score'] / (df_fe['num_late_payments'] + 1)
    
    # Encode categorical variables
    le_age = LabelEncoder()
    le_income = LabelEncoder()
    
    df_fe['age_group_encoded'] = le_age.fit_transform(df_fe['age_group'])
    df_fe['income_category_encoded'] = le_income.fit_transform(df_fe['income_category'])
    
    # Select features for modeling
    feature_columns = [
        'age', 'income', 'debt_to_income_ratio', 'credit_history_length',
        'num_credit_accounts', 'payment_history_score', 'credit_utilization',
        'num_late_payments', 'employment_length', 'loan_amount',
        'income_to_loan_ratio', 'debt_burden', 'credit_experience',
        'payment_reliability', 'age_group_encoded', 'income_category_encoded'
    ]
    
    print(f"Number of features after engineering: {len(feature_columns)}")
    print("New features created:")
    for col in ['income_to_loan_ratio', 'debt_burden', 'credit_experience', 'payment_reliability']:
        print(f"- {col}")
    
    return df_fe, feature_columns

# Apply feature engineering
df_engineered, features = feature_engineering(df_analyzed)


def preprocess_data(df, feature_columns, target_column='creditworthy'):
    """
    Prepare data for machine learning
    """
    print("=== DATA PREPROCESSING ===")
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Target distribution in training set:")
    print(y_train.value_counts(normalize=True))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Preprocess the data
X_train, X_test, y_train, y_test, scaler = preprocess_data(df_engineered, features)


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models and evaluate their performance
    """
    print("=== MODEL TRAINING AND EVALUATION ===")
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    return results

# Train and evaluate all models
model_results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Summary of results
print("\n=== MODEL COMPARISON SUMMARY ===")
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
print("-" * 80)
for name, metrics in model_results.items():
    print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['roc_auc']:<10.4f}")

# Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
print(f"\nBest performing model: {best_model_name} (ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f})")

# Feature importance for Random Forest
if 'Random Forest' in model_results:
    rf_model = model_results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== FEATURE IMPORTANCE (Random Forest) ===")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importance - Random Forest')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

print("\n=== CREDIT SCORING SYSTEM ANALYSIS COMPLETE ===")