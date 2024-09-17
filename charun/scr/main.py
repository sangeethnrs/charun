from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from data_utils import load_and_preprocess_data
from model import train_random_forest
import pandas as pd
import numpy as np

app = FastAPI()

class DataInput(BaseModel):
    data: list

@app.post("/predict/")
def predict(input_data: DataInput, dataset: str = Query(...)):
    try:
        # Load and preprocess data based on selected dataset
        X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset, dataset_paths[dataset])

        # Train model
        model = train_random_forest(X_train, y_train)
        
        # Convert input data to DataFrame for consistency with model input
        input_df = pd.DataFrame([input_data.data], columns=X_test.columns)

        # Check if the input DataFrame has the same columns as the training data
        if not np.all(input_df.columns == X_test.columns):
            raise HTTPException(status_code=400, detail="Input data features do not match the model's expected features.")
        
        # Make a prediction
        prediction = model.predict(input_df)
        
        return {"prediction": int(prediction[0])}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Paths to the datasets
dataset_paths = {
    "Bank_churn": r"C:\Users\Administrator\Desktop\charun\churn_data\Bank_churn.csv",
    "BankChurners": r"C:\Users\Administrator\Desktop\charun\churn_data\BankChurners.csv",
    "churn_bigml_80": r"C:\Users\Administrator\Desktop\charun\churn_data\churn-bigml-80.csv",
    "Customertravel": r"C:\Users\Administrator\Desktop\charun\churn_data\Customertravel.csv",
    "E_Commerce": r"C:\Users\Administrator\Desktop\charun\churn_data\E Commerce Dataset.csv",
    "ecom_user_churn": r"C:\Users\Administrator\Desktop\charun\churn_data\ecom-user-churn-data.csv"
}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
