from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import uvicorn
from data_models import PredictionDataset
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="NYC Taxi Prediction API")

current_file_path = Path(__file__).parent
model_path = current_file_path / "models" / "models" / "xgbreg.joblib"
preprocessor_path = current_file_path / "models" / "transformers" / "preprocessor.joblib" 
output_transformer_path = current_file_path / "models" / "transformers" / "output_transformer.joblib"

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)
output_transformer = joblib.load(output_transformer_path)

model_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", model)
])



@app.get("/")
def home():
    return "Welcome to taxi price prediction app"

@app.post("/predictions")
def do_predictions(test_data: PredictionDataset):
    X_test = pd.DataFrame(
        data = {
            'vendor_id':test_data.vendor_id,
            'passenger_count':test_data.passenger_count,
            'pickup_longitude':test_data.pickup_longitude,
            'pickup_latitude':test_data.pickup_latitude,
            'dropoff_longitude':test_data.dropoff_longitude,
            'dropoff_latitude':test_data.dropoff_latitude,
            'pickup_hour':test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month':test_data.pickup_month,
            'pickup_day':test_data.pickup_day,
            'is_weekend':test_data.is_weekend,
            'haversine_distance':test_data.haversine_distance,
            'euclidean_distance':test_data.euclidean_distance,
            'manhattan_distance':test_data.manhattan_distance
         }, index=[0]
    )
    predictions = model_pipeline.predict(X_test).reshape(-1, 1)
    output_inverse_trnsfrm = output_transformer.inverse_transform(predictions)[0].item()
    return f"Trip duration for the trip is {output_inverse_trnsfrm:.2f} minutes"



if __name__ == "__main__":
    uvicorn.run(app="app:app", host="127.0.0.1", port=8000)
   
    
    
#get -> get some response from API

#post -> sending something to API

#put -> updating something to API

#delete -> deleting something from API