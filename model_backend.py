# model_backend.py
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data():
    features = pd.read_csv("features.csv", parse_dates=['Date'])
    features_reduced = features.iloc[::3, :]
    stores = pd.read_csv("stores.csv")
    train = pd.read_csv("train.csv", parse_dates=['Date'])
    train_reduced = train.iloc[::12, :]
    merged_df = train_reduced.merge(features_reduced, on='Store', how='left').merge(stores, on='Store', how='left')
    print(merged_df.head())
    for col in merged_df.columns:
        print(col)
    return merged_df

def preprocess_data(df):
    df['Date_x'] = pd.to_datetime(df['Date_x'])
    df.fillna(0, inplace=True)
    return df

def train_test_data_split(df, test_size=0.25):
    # Ensuring the dataframe is sorted by date before splitting
    df = df.sort_values(by='Date_x')
    return train_test_split(df, test_size=test_size, shuffle=False)

def train_prophet_model(df):
    print("start training")
    prophet_df = df[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(prophet_df)
    return model

def predict_prophet(model, periods=52):
    future = model.make_future_dataframe(periods=periods, freq='W')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def predict_for_date(model, date_str):
    """
    Predicts weekly sales for the week of the given date.
    :param model: The trained Prophet model.
    :param date_str: The date string in 'YYYY-MM-DD' format.
    :return: A dictionary with the prediction and intervals.
    """
    future_date = pd.to_datetime(date_str)
    future = pd.DataFrame([future_date], columns=['ds'])
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[0].to_dict()

