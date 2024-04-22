import pandas as pd
import model_backend
from prophet import Prophet
import unittest
from httpx import AsyncClient
from main_api import app

class ModelBackendTest(unittest.TestCase):
    def create_mock_dataframe(self):
        data = {
            'Store': [1, 1, 2, 2],
            'Dept': [1, 1, 2, 2],
            'Date_x': pd.date_range(start='2022-01-01', periods=4),
            'Weekly_Sales': [100.0, 200.0, 300.0, 400.0],
            'IsHoliday_x': [False, True, False, True],
            'Date_y': pd.date_range(start='2022-01-01', periods=4),
            'Temperature': [55, 60, 65, 70],
            'Fuel_Price': [2.5, 2.6, 2.7, 2.8],
            'MarkDown1': [0.0, 0.0, 100.0, 200.0],
            'MarkDown2': [0.0, 0.0, 50.0, 100.0],
            'MarkDown3': [0.0, 0.0, 20.0, 40.0],
            'MarkDown4': [0.0, 0.0, 30.0, 60.0],
            'MarkDown5': [0.0, 0.0, 40.0, 80.0],
            'CPI': [220.0, 221.0, 222.0, 223.0],
            'Unemployment': [5.0, 5.1, 5.2, 5.3],
            'IsHoliday_y': [False, True, False, True],
            'Type': ['A', 'A', 'B', 'B'],
            'Size': [200000, 200000, 150000, 150000]
        }
        mock_df = pd.DataFrame(data)
        return mock_df

    def test_preprocess_data(self):
        df = self.create_mock_dataframe()
        df.loc[2, 'Weekly_Sales'] = None
        
        processed_df = model_backend.preprocess_data(df)
        assert processed_df.isnull().sum().sum() == 0, "All NaN values should be filled."


    def test_train_test_data_split(self):
        df = self.create_mock_dataframe().sort_values(by='Date_x')
        train, test = model_backend.train_test_data_split(df, test_size=0.25)
        assert len(train) == 3, "Train set size should be 3."
        assert len(test) == 1, "Test set size should be 1."
        assert train['Date_x'].is_monotonic_increasing, "Train data should be sorted by Date_x."
        assert test['Date_x'].is_monotonic_increasing, "Test data should be sorted by Date_x."

    def test_predict_prophet(self):
        # Mock model setup omitted for brevity
        model = Prophet().fit(pd.DataFrame({
            'ds': pd.date_range(start='2022-01-01', periods=10),
            'y': range(10)
        }))
        forecast = model_backend.predict_prophet(model, periods=10)
        assert set(forecast.columns) == {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
        print(len(forecast))
        assert len(forecast) == 20, "Forecast should contain 20 periods."

    def test_predict_for_date(self):
        model = Prophet().fit(self.create_mock_dataframe()[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'}))
        result = model_backend.predict_for_date(model, '2022-01-05')
        expected_keys = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}
        assert set(result.keys()) == expected_keys, "Result should contain the expected keys."
    
class TestAPI(unittest.TestCase):
    async def test_root_endpoint(self):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/")
            assert response.status_code == 200
            assert response.json() == {"message": "Hello World"}

    async def test_predict_sales(self):
        test_date = {"date": "2022-01-05"}
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=test_date)
            assert response.status_code == 200
            assert "yhat" in response.json()
    
    async def test_invalid_prediction_payload(client):
        response = await client.post("/predict", json={"date": "not-a-date"})
        assert response.status_code == 422  

    async def test_boundary_prediction_payload(client):
        response = await client.post("/predict", json={"date": "1900-01-01"})
        assert response.status_code == 200  

    async def test_injection_attempt(client):
        response = await client.post("/predict", json={"date": "2022-01-01; DROP TABLE USERS"})
        assert response.status_code == 422

    async def test_preprocess_data_with_outliers(self):
        df = self.create_mock_dataframe()
        df['Weekly_Sales'] = [1000000, 200, 300, -400]  
        processed_df = model_backend.preprocess_data(df)
        assert processed_df['Weekly_Sales'].max() <= 300000, "Outliers should be capped."

    async def test_predict_with_negative_values(self):
        df = self.create_mock_dataframe()
        df['Weekly_Sales'] = [100, -200, 300, 400]  
        model = Prophet().fit(df[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'}))
        result = model_backend.predict_for_date(model, '2022-01-05')
        print(result)
        assert 'yhat' in result, "Prediction should still occur with negative values."



