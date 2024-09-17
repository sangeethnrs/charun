import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

def load_and_preprocess_data(dataset_name, dataset_path):
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Display the first few rows and columns of the dataset
    print(f"Processing dataset: {dataset_name}")
    print(data.head())
    print("Columns in the dataset:")
    print(data.columns.tolist())

    # Define target and feature columns based on dataset
    if dataset_name == "Bank_churn":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col, 'rownumber', 'customerid', 'surname'])
        numeric_features = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary']
        categorical_features = ['geography', 'gender']
        
    elif dataset_name == "BankChurners":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col, 'clientnum'])
        numeric_features = [
            'customer_age', 'dependent_count', 'months_on_book', 'total_relationship_count',
            'months_inactive_12_mon', 'contacts_count_12_mon', 'credit_limit', 'total_revolving_bal',
            'avg_open_to_buy', 'total_amt_chng_q4_q1', 'total_trans_amt', 'total_trans_ct',
            'total_ct_chng_q4_q1', 'avg_utilization_ratio'
        ]
        categorical_features = [
            'gender', 'education_level', 'marital_status', 'income_category', 'card_category'
        ]

    elif dataset_name == "churn_bigml_80":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col])
        numeric_features = [
            'account length', 'total day minutes', 'total day calls', 'total day charge', 
            'total eve minutes', 'total eve calls', 'total eve charge', 
            'total night minutes', 'total night calls', 'total night charge', 
            'total intl minutes', 'total intl calls', 'total intl charge', 
            'customer service calls'
        ]
        categorical_features = [
            'state', 'international plan', 'voice mail plan'
        ]

    elif dataset_name == "Customertravel":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col])
        numeric_features = ['age']
        categorical_features = [
            'frequentflyer', 'annualincomeclass', 'servicesopted', 
            'accountsyncedtosocialmedia', 'bookedhotelornot'
        ]

    elif dataset_name == "E_Commerce":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col, 'customerid'])
        numeric_features = [
            'tenure', 'hourspendonapp', 'numberofdeviceregistered', 'satisfactionscore',
            'numberofaddress', 'orderamounthikefromlastyear', 'ordercount', 'daysincelastorder',
            'cashbackamount'
        ]
        categorical_features = [
            'preferredlogindevice', 'citytier', 'warehousetohome', 'preferredpaymentmode',
            'gender', 'preferedordercat', 'maritalstatus', 'complain', 'couponused'
        ]

    elif dataset_name == "ecom_user_churn":
        target_col = 'churn'
        feature_cols = data.columns.difference([target_col])
        numeric_features = [
            'ses_rec', 'ses_rec_avg', 'ses_rec_sd', 'ses_rec_cv', 'user_rec', 'ses_n', 
            'ses_n_r', 'int_n', 'int_n_r', 'tran_n', 'tran_n_r', 'rev_sum', 'rev_sum_r',
            'major_spend_r', 'int_cat_n_avg', 'int_itm_n_avg', 'ses_mo_avg', 'ses_mo_sd',
            'ses_ho_avg', 'ses_ho_sd', 'ses_wknd_r', 'ses_len_avg', 'time_to_int', 'time_to_tran'
        ]
        categorical_features = [
            'int_cat1_n', 'int_cat2_n', 'int_cat3_n', 'int_cat4_n', 'int_cat5_n', 
            'int_cat6_n', 'int_cat7_n', 'int_cat8_n', 'int_cat9_n', 'int_cat10_n', 
            'int_cat11_n', 'int_cat12_n', 'int_cat13_n', 'int_cat15_n', 'int_cat16_n', 
            'int_cat17_n', 'int_cat18_n', 'int_cat19_n', 'int_cat20_n', 'int_cat21_n', 
            'int_cat22_n', 'int_cat23_n', 'int_cat24_n'
        ]

    else:
        raise ValueError("Unknown dataset name")

    # Check for missing columns in the dataset
    missing_numeric_features = [col for col in numeric_features if col not in data.columns]
    missing_categorical_features = [col for col in categorical_features if col not in data.columns]
    
    if missing_numeric_features:
        print(f"Warning: Missing numeric features: {missing_numeric_features}")
    if missing_categorical_features:
        print(f"Warning: Missing categorical features: {missing_categorical_features}")

    # Adjust feature columns based on missing columns
    numeric_features = [col for col in numeric_features if col in data.columns]
    categorical_features = [col for col in categorical_features if col in data.columns]

    # Split features and target
    X = data[feature_cols]
    y = data[target_col]

    # Define preprocessing pipelines
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
        ('scaler', StandardScaler())  # Standardize features
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value
        ('onehot', OneHotEncoder(drop='first'))  # One hot encode categorical features
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Preprocessing and feature engineering
    X_processed = preprocessor.fit_transform(X)

    # Feature interactions
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X_processed)

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_interactions, y, test_size=0.2, random_state=42)

    # Display the shapes of the resulting datasets
    print(f"Dataset: {dataset_name}")
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Training target shape:", y_train.shape)
    print("Testing target shape:", y_test.shape)
    
    return X_train, X_test, y_train, y_test
