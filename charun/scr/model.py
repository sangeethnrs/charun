from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_random_forest(X_train, y_train):
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report

if __name__ == "__main__":
    from data_utils import load_and_preprocess_data
    
    # Define dataset paths
    dataset_paths = {
        "Bank_churn": r"C:\Users\Administrator\Desktop\charun\churn_data\Bank_churn.csv",
        "BankChurners": r"C:\Users\Administrator\Desktop\charun\churn_data\BankChurners.csv",
        "churn_bigml_80": r"C:\Users\Administrator\Desktop\charun\churn_data\churn-bigml-80.csv",
        "Customertravel": r"C:\Users\Administrator\Desktop\charun\churn_data\Customertravel.csv",
        "E_Commerce": r"C:\Users\Administrator\Desktop\charun\churn_data\E Commerce Dataset.csv"
    }
    
    # Process and train models for each dataset
    for dataset_name, dataset_path in dataset_paths.items():
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_name, dataset_path)
        
        # Train the model
        model = train_random_forest(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
