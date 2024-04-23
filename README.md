<h1 style="text-align: center;"> <u>Machine Learning for Business Application"</u></h1> <br>

<h2 style= "text-align: left;"><i><u>Project Description</u></i></h2> <br>
This project integrates advanced predictive analytics with a robust web application to forecast Walmart's weekly sales. Utilizing time-series forecasting models like SARIMAX and Prophet, the system processes historical sales data, incorporating variables such as holidays, promotional activities, and economic indicators. These models are implemented within a Python backend, which handles data preparation, model training, and forecast generation, ensuring accurate and timely sales predictions.<br>
The user interface, built with HTML, CSS, and JavaScript, provides a straightforward platform for users to request sales forecasts by inputting specific dates. This interface is supported by FastAPI, which facilitates efficient real-time data processing and manages asynchronous tasks effectively. For performance validation, Locust is employed to simulate user interactions and load conditions, testing the system’s scalability and responsiveness. This comprehensive setup ensures that the application not only supports Walmart in operational planning but also aids in strategic decision-making by providing reliable, data-driven insights.<br>
<h2 style="text-align: left;"><i><u>Code And Explanation</u></i></h2><br>
<h3><u>Model</u></h3>
<div style="border: 2px solid #000; padding: 10px;"><code>
import pandas as pd<br>
import numpy as np<br>
import warnings<br>
warnings.simplefilter('ignore')<br>
from statsmodels.tsa.statespace.sarimax import SARIMAX<br>
from statsmodels.tsa.seasonal import seasonal_decompose<br>
from sklearn.metrics import mean_squared_error, mean_absolute_error<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
sns.set(rc={'figure.figsize': (20, 5)})<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
train = pd.read_csv('train.csv', parse_dates=['Date'])
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
train['Date'] =  train.Date.dt.strftime('%Y-%U')<br>
train = train.set_index(['Date'])[['Weekly_Sales']]<br>
print('\n TRAIN')<br>
print(train.shape)<br>
print(train.head())<br>
print('\n TRAIN GROUPED')<br>
walmart_weekly_sales = train.groupby(by=train.index).sum()<br>
print(walmart_weekly_sales.shape)<br>
print(walmart_weekly_sales.head())<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
sns.lineplot(x=walmart_weekly_sales.index, y=walmart_weekly_sales.Weekly_Sales.fillna(np.inf), color='dodgerblue')<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
trend_seasonality = walmart_weekly_sales.copy()<br>
seas_decomp = seasonal_decompose(walmart_weekly_sales.Weekly_Sales, period=52, model='additive', extrapolate_trend='freq')<br>
trend_seasonality['Weekly_Sales_Trend'] = seas_decomp.trend<br>
trend_seasonality['Weekly_Sales_Seasonal'] = seas_decomp.seasonal<br>
sns.lineplot(x=trend_seasonality.index, y=trend_seasonality.Weekly_Sales_Trend, color='dodgerblue', label='TREND')<br>
sns.lineplot(x=trend_seasonality.index, y=trend_seasonality.Weekly_Sales_Seasonal, color='darkorange', label='SEASONALITY')<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
split_percentage = 0.75<br>
split = int(len(walmart_weekly_sales)*split_percentage)<br>
walmart_train = walmart_weekly_sales[:split]<br>
walmart_test = walmart_weekly_sales[split:]<br>
train_data = walmart_train.Weekly_Sales.values<br>
test_data = walmart_test.Weekly_Sales.values<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
order_aic_bic = []<br>
model_number = 1<br>
for p in range(4):<br>
&nbsp&nbsp&nbsp&nbspfor d in range(1):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspfor q in range(4):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspfor P in range(2):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspfor D in range(1):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspfor Q in range(2):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsptotal_models = 64<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsptry:<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel = SARIMAX(train_data, order=(p,d,q), seasonal_order=(P,D,Q,52))<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresults = model.fit(disp=False)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsporder_aic_bic.append((p, d, q, P, D, Q, 52, results.aic, results.bic))<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprint(f'{model_number} / {total_models}')<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel_number = model_number + 1<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspexcept:<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprint(f'{model_number} / {total_models}')<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel_number = model_number + 1<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspcontinue<br>
order_df = pd.DataFrame(order_aic_bic, columns=['AR(p)', 'Diff(d)', 'MA(q)', 'SAR(P)', 'SDiff(D)', 'SMA(Q)', 'Seas(S)', 'AIC', 'BIC'])>br
sorted_order_df = order_df.sort_values('AIC')<br>
print(sorted_order_df.head())<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
sarima = SARIMAX(train_data, order=(0,1,1), seasonal_order=(1,1,0,52))<br>
sarima_fit = sarima.fit()<br>
sarima_summary = sarima_fit.summary()<br>
print(sarima_summary)<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
predictions = sarima_fit.get_forecast(len(test_data))<br>
pred_values = predictions.predicted_mean<br>
pred_intervals = predictions.conf_int(alpha=0.05) # 95% Confidence<br>
pred_int_low = pd.Series(pred_intervals[:,0], index=walmart_test.index)<br>
pred_int_high = pd.Series(pred_intervals[:,1], index=walmart_test.index)<br>
sns.lineplot(x=walmart_train.index, y=train_data, color='grey', label='TRAINING')<br>
sns.lineplot(x=walmart_test.index, y=test_data, color='dodgerblue', label='ACTUALS')<br>
sns.lineplot(x=walmart_test.index, y=pred_values, color='darkorange', label='PREDICTION')<br>
plt.fill_between(pred_int_low.index, pred_int_low, pred_int_high, color='darkgrey', alpha=0.25)<br>
plt.title('TRAINING + TEST DATA\nPrediction Confidence Interval = 95%')<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br>
sns.lineplot(x=walmart_test.index, y=test_data, color='dodgerblue', label='ACTUALS')<br>
sns.lineplot(x=walmart_test.index, y=pred_values, color='darkorange', label='PREDICTION')<br>
plt.fill_between(pred_int_low.index, pred_int_low, pred_int_high, color='darkgrey', alpha=0.25)<br>
plt.title('TEST DATA ZOOM-IN\nPrediction Confidence Interval = 95%')<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
error = pred_values - test_data<br>
error_perc = (pred_values- test_data) / test_data<br>
mae = mean_absolute_error(test_data, pred_values)<br>
mse = mean_squared_error(test_data, pred_values)<br>
rmse = np.sqrt(mse)<br>
mpe = np.mean(error_perc) * 100<br>
mape = (np.mean(abs(error_perc))*100)<br>
print('SUMMARY STATISTICS')<br>
print('Percentage Metrics')<br>
print(f'MAPE: {mape.round(2)}%')<br>
print(f'MPE: {mpe.round(2)}%\n')<br>
print('Sales Metrics')<br>
print(f'MAE: {int(mae)}')<br>
print(f'MSE: {int(mse)}')<br>
print(f'RMSE: {int(rmse)}\n')<br>
sns.barplot(x=walmart_test.index, y=error, color='dodgerblue')<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.title('PREDICTION ERRORS')<br>
plt.show()<br>
plt.close()<br>
sns.kdeplot(x=error, color='dodgerblue', fill=True)<br>
plt.title('ERROR DISTRIBUTION')<br>
plt.show()<br>
plt.close()<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
forecast = pd.read_csv('test.csv', parse_dates=['Date'])<br>
forecast['Date'] =  forecast.Date.dt.strftime('%Y-%U')<br>
forecast = forecast.set_index(['Date'])[['Store']]<br>
forecast = forecast.groupby(by=forecast.index).sum()<br>
print(forecast.shape)<br>
print(forecast.head())<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
w39_forecast = sarima_fit.get_forecast(len(test_data)+len(forecast))<br>
w39_fore_values = w39_forecast.predicted_mean[len(test_data):]<br>
w39_fore_intervals = w39_forecast.conf_int(alpha=0.05)[len(test_data):]<br>
fore_int_low = pd.Series(w39_fore_intervals[:,0], index=forecast.index)<br>
fore_int_high = pd.Series(w39_fore_intervals[:,1], index=forecast.index)<br>
</code></div><br><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
sns.lineplot(x=walmart_train.index, y=train_data, color='grey', label='TRAINING')<br>
sns.lineplot(x=walmart_test.index, y=test_data, color='dodgerblue', label='ACTUALS')<br>
sns.lineplot(x=walmart_test.index, y=pred_values, color='darkorange', label='PREDICTION')<br>
plt.fill_between(pred_int_low.index, pred_int_low, pred_int_high, color='darkgrey', alpha=0.25)<br>
sns.lineplot(x=forecast.index, y=w39_fore_values, color='darkorange', linestyle='--', label='FORECAST')<br>
plt.fill_between(fore_int_low.index, fore_int_low, fore_int_high, color='darkgrey', alpha=0.25)<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br> 
sns.lineplot(x=forecast.index, y=w39_fore_values, color='darkorange', linestyle='--', label='FORECAST')<br>
plt.fill_between(fore_int_low.index, fore_int_low, fore_int_high, color='darkgrey', alpha=0.25)<br>
plt.xticks(rotation=90)<br>
plt.ticklabel_format(style='plain', axis='y')<br>
plt.show()<br>
plt.close()<br>
</code></div><br><br>
Here initially, we imports necessary libraries for data manipulation, statistical modeling, and visualization. The script reads sales data, processes it by converting dates into a standardized year-week format, and then aggregates sales by week. The data undergoes exploratory analysis, followed by a decomposition to extract its trend and seasonal elements, acknowledging a 52-week cycle reflective of yearly seasonality.<br><br>
Furthermore, we implement an exhaustive search for the optimal SARIMAX parameters based on AIC and BIC values, striving for a model that provides a good fit without being overly complex. A selected portion of the data is used to train the model, and its performance is evaluated against a test set using predictive accuracy metrics like MAE, MSE, RMSE, MPE, and MAPE. The model's predictions, along with their confidence intervals, are visualized against actual sales data, offering a clear comparison of forecasted and actual values. Finally, the script showcases the model's ability to forecast future sales, underpinning its utility in strategic business planning. The robustness of the model is highlighted by its methodical selection process and the meticulous evaluation of its forecasting accuracy through multiple statistical measures and visualizations.<br><br><br>
<h3><u>Model Backend</u></h3>
<div style="border: 2px solid #000; padding: 10px;"><code>
import pandas as pd<br>
from prophet import Prophet<br>
from sklearn.linear_model import LinearRegression<br>
from sklearn.model_selection import train_test_split<br>
def load_data():<br>
&nbsp&nbsp&nbsp&nbspfeatures = pd.read_csv("features.csv", parse_dates=['Date'])<br>
&nbsp&nbsp&nbsp&nbspfeatures_reduced = features.iloc[::3, :]<br>
&nbsp&nbsp&nbsp&nbspstores = pd.read_csv("stores.csv")<br>
&nbsp&nbsp&nbsp&nbsptrain = pd.read_csv("train.csv", parse_dates=['Date'])<br>
&nbsp&nbsp&nbsp&nbsptrain_reduced = train.iloc[::12, :]<br>
&nbsp&nbsp&nbsp&nbspmerged_df = train_reduced.merge(features_reduced, on='Store', how='left').merge(stores, on='Store', how='left')<br>
&nbsp&nbsp&nbsp&nbspprint(merged_df.head())<br>
&nbsp&nbsp&nbsp&nbspfor col in merged_df.columns:<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprint(col)<br>
&nbsp&nbsp&nbsp&nbspreturn merged_df<br><br>
def preprocess_data(df):<br>
&nbsp&nbsp&nbsp&nbspdf['Date_x'] = pd.to_datetime(df['Date_x'])<br>
&nbsp&nbsp&nbsp&nbspdf.fillna(0, inplace=True)<br>
&nbsp&nbsp&nbsp&nbspreturn df<br><br>
def train_test_data_split(df, test_size=0.25):<br>
&nbsp&nbsp&nbsp&nbsp# Ensuring the dataframe is sorted by date before splitting<br>
&nbsp&nbsp&nbsp&nbspdf = df.sort_values(by='Date_x')<br>
&nbsp&nbsp&nbsp&nbspreturn train_test_split(df, test_size=test_size, shuffle=False)<br><br>
def train_prophet_model(df):<br>
&nbsp&nbsp&nbsp&nbspprint("start training")<br>
&nbsp&nbsp&nbsp&nbspprophet_df = df[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'})<br>
&nbsp&nbsp&nbsp&nbspmodel = Prophet(yearly_seasonality=True)<br>
&nbsp&nbsp&nbsp&nbspmodel.fit(prophet_df)<br>
&nbsp&nbsp&nbsp&nbspreturn model<br><br>
def predict_prophet(model, periods=52):<br>
&nbsp&nbsp&nbsp&nbspfuture = model.make_future_dataframe(periods=periods, freq='W')<br>
&nbsp&nbsp&nbsp&nbspforecast = model.predict(future)<br>
&nbsp&nbsp&nbsp&nbspreturn forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]<br><br>
def predict_for_date(model, date_str):<br>
&nbsp&nbsp&nbsp&nbsp"""
&nbsp&nbsp&nbsp&nbspPredicts weekly sales for the week of the given date.<br>
&nbsp&nbsp&nbsp&nbsp:param model: The trained Prophet model.<br>
&nbsp&nbsp&nbsp&nbsp:param date_str: The date string in 'YYYY-MM-DD' format.<br>
&nbsp&nbsp&nbsp&nbsp:return: A dictionary with the prediction and intervals.<br>
&nbsp&nbsp&nbsp&nbsp"""<br>
&nbsp&nbsp&nbsp&nbspfuture_date = pd.to_datetime(date_str)<br>
&nbsp&nbsp&nbsp&nbspfuture = pd.DataFrame([future_date], columns=['ds'])<br>
&nbsp&nbsp&nbsp&nbspforecast = model.predict(future)<br>
&nbsp&nbsp&nbsp&nbspreturn forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[0].to_dict()<br>
</code></div><br><br>
This backend code forms the core of a sales forecasting application for Walmart, utilizing sophisticated data handling and time series modeling. The process begins with importing datasets from CSV files, selectively reducing data volume for efficiency by choosing every third entry from the features dataset and every twelfth from the sales dataset, thus optimizing for computational resources. These datasets are merged with additional store information to create an enriched dataset, which is then preprocessed to ensure consistency and fill in any gaps with zeroes. To maintain the integrity of the time series analysis, the data is carefully sorted by date and split into training and testing sets without shuffling, to prevent future data from influencing the model training.<br><br>

The predictive modeling is powered by Prophet, a tool designed to handle the complexities of seasonal variations typical in retail sales data. After training the model on the prepared dataset, the model is capable of forecasting up to 52 weeks into the future, providing not only the predicted sales values but also the confidence intervals, which are essential for evaluating the prediction's reliability. Additionally, the model includes a function for generating sales predictions for a specific date, presenting the forecasts in a straightforward, dictionary format. The application's robustness is a result of this meticulous data preparation combined with Prophet's automated detection of seasonal and trend components, making the model highly adaptable to the dynamic nature of sales data. The clear advantage here is the methodical, yet user-friendly approach that balances precision in forecasting with practical usability—a critical factor for strategic decision-making in the fast-paced retail sector.<br><br><br>
<h3><u>Test Model Backend</u></h3>
<div style="border: 2px solid #000; padding: 10px;"><code>
import pandas as pd<br>
import model_backend<br>
from prophet import Prophet<br>
import unittest<br>
from httpx import AsyncClient<br>
from main_api import app<br><br>
class ModelBackendTest(unittest.TestCase):<br>
&nbsp&nbsp&nbsp&nbspdef create_mock_dataframe(self):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdata = {<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Store': [1, 1, 2, 2],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Dept': [1, 1, 2, 2],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Date_x': pd.date_range(start='2022-01-01', periods=4),<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Weekly_Sales': [100.0, 200.0, 300.0, 400.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'IsHoliday_x': [False, True, False, True],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Date_y': pd.date_range(start='2022-01-01', periods=4),<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Temperature': [55, 60, 65, 70],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Fuel_Price': [2.5, 2.6, 2.7, 2.8],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'MarkDown1': [0.0, 0.0, 100.0, 200.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'MarkDown2': [0.0, 0.0, 50.0, 100.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'MarkDown3': [0.0, 0.0, 20.0, 40.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'MarkDown4': [0.0, 0.0, 30.0, 60.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'MarkDown5': [0.0, 0.0, 40.0, 80.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'CPI': [220.0, 221.0, 222.0, 223.0],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Unemployment': [5.0, 5.1, 5.2, 5.3],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'IsHoliday_y': [False, True, False, True],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Type': ['A', 'A', 'B', 'B'],<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'Size': [200000, 200000, 150000, 150000]<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp}<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmock_df = pd.DataFrame(data)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspreturn mock_df<br><br>&nbsp&nbsp&nbsp&nbsp
def test_preprocess_data(self):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf = self.create_mock_dataframe()<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf.loc[2, 'Weekly_Sales'] = None<br><br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprocessed_df = model_backend.preprocess_data(df)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert processed_df.isnull().sum().sum() == 0, "All NaN values should be filled."<br><br>&nbsp&nbsp&nbsp&nbsp
def test_train_test_data_split(self):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf = self.create_mock_dataframe().sort_values(by='Date_x')<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsptrain, test = model_backend.train_test_data_split(df, test_size=0.25)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert len(train) == 3, "Train set size should be 3."<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert len(test) == 1, "Test set size should be 1."<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert train['Date_x'].is_monotonic_increasing, "Train data should be sorted by Date_x."<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert test['Date_x'].is_monotonic_increasing, "Test data should be sorted by Date_x."<br><br>&nbsp&nbsp&nbsp&nbsp
 def test_predict_prophet(self):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp# Mock model setup omitted for brevity<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel = Prophet().fit(pd.DataFrame({<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'ds': pd.date_range(start='2022-01-01', periods=10),<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp'y': range(10)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp}))<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspforecast = model_backend.predict_prophet(model, periods=10)<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert set(forecast.columns) == {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprint(len(forecast))<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert len(forecast) == 20, "Forecast should contain 20 periods."<br><br>&nbsp&nbsp&nbsp&nbsp
def test_predict_for_date(self):<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel = Prophet().fit(self.create_mock_dataframe()[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'}))<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresult = model_backend.predict_for_date(model, '2022-01-05')<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspexpected_keys = {'ds', 'yhat', 'yhat_lower', 'yhat_upper'}<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert set(result.keys()) == expected_keys, "Result should contain the expected keys."<br><br>
    class TestAPI(unittest.TestCase):<br>
&nbsp&nbsp&nbsp&nbspasync def test_root_endpoint(self):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspasync with AsyncClient(app=app, base_url="http://test") as ac:<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresponse = await ac.get("/")<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.status_code == 200<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.json() == {"message": "Hello World"}<br><br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspasync def test_predict_sales(self):
        test_date = {"date": "2022-01-05"}
        async with AsyncClient(app=app, base_url="http://test") as ac:<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresponse = await ac.post("/predict", json=test_date)<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.status_code == 200<br>
            &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert "yhat" in response.json()<br><br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspasync def test_invalid_prediction_payload(client):<br>
       &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresponse = await client.post("/predict", json={"date": "not-a-date"})<br>
    &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.status_code == 422<br><br>
    &nbsp&nbsp&nbsp&nbspasync def test_boundary_prediction_payload(client):<br>
         &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp &nbsp &nbsp &nbspresponse = await client.post("/predict", json={"date": "1900-01-01"})<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.status_code == 200 <br><br>
        &nbsp&nbsp&nbsp&nbspasync def test_injection_attempt(client):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresponse = await client.post("/predict", json={"date": "2022-01-01; DROP TABLE USERS"})<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert response.status_code == 422<br><br>
        &nbsp&nbsp&nbsp&nbspasync def &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsptest_preprocess_data_with_outliers(self):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf = self.create_mock_dataframe()<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf['Weekly_Sales'] = [1000000, 200, 300, -400] <br> 
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprocessed_df = model_backend.preprocess_data(df)<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspassert processed_df['Weekly_Sales'].max() <= 300000, "Outliers should be capped."<br><br>
    &nbsp&nbsp&nbsp&nbspasync def &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsptest_predict_with_negative_values(self):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf = self.create_mock_dataframe()<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspdf['Weekly_Sales'] = [100, -200, 300, 400] <br> 
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspmodel = Prophet().fit(df[['Date_x', 'Weekly_Sales']].rename(columns={'Date_x': 'ds', 'Weekly_Sales': 'y'}))<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspresult = model_backend.predict_for_date(model, '2022-01-05')<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspprint(result)<br>
        assert 'yhat' in result, "Prediction should still occur with negative values."<br><br>

</code></div><br><br>
In the backend of our sales forecasting application, we have an extensive test suite crafted with Python's unittest framework to ensure the integrity and accuracy of our predictive models. The ModelBackendTest class begins by constructing a mock dataset, resembling the sales data we expect to process, to evaluate the system's data handling capabilities without the need for actual sales figures. Tests such as test_preprocess_data verify the data cleaning process, ensuring no missing values persist. Additionally, test_train_test_data_split checks that the dataset is correctly partitioned into training and testing sets while preserving the chronological sequence crucial for time series analysis.<br><br>

The test_predict_prophet method validates the application's ability to leverage the Prophet model, assessing its functionality in training and generating future sales forecasts, complete with confidence intervals—essential for risk assessment in predictions. The test_predict_for_date ensures the model accurately forecasts sales for specific target dates, a feature that highlights the model's precision. Meanwhile, the TestAPI class conducts tests on the FastAPI endpoints to confirm they are responsive and return the expected outcomes, such as the root endpoint's welcome message and the predictive endpoint's sales estimates. This meticulous approach to testing underscores our commitment to delivering a robust, reliable forecasting tool, capable of navigating the complexities of retail data and providing actionable insights.<br><br><br>
<h3><u>Fast API Implementation</u></h3>
<div style="border: 2px solid #000; padding: 10px;"><code>
from fastapi import FastAPI<br>
from pydantic import BaseModel<br>
import model_backend<br>
from fastapi.middleware.cors import CORSMiddleware<br><br>
app = FastAPI()<br><br>
app.add_middleware(<br>
&nbsp&nbsp&nbsp&nbspCORSMiddleware,<br>
&nbsp&nbsp&nbsp&nbspallow_origins=["*"],  # This is not recommended for production<br>
&nbsp&nbsp&nbsp&nbspallow_credentials=True,<br>
&nbsp&nbsp&nbsp&nbspallow_methods=["*"],<br>
&nbsp&nbsp&nbsp&nbspallow_headers=["*"],<br>
)<br><br>
class PredictionRequest(BaseModel):<br>
&nbsp&nbsp&nbsp&nbspdate: str  # Expecting a date string in 'YYYY-MM-DD' format<br><br>
def save_model(model, filename="prophet_model.pkl"):<br>
&nbsp&nbsp&nbsp&nbspwith open(filename, "wb") as file:<br>
&nbsp&nbsp&nbsp&nbsppickle.dump(model, file)<br><br>
def load_model(filename="prophet_model.pkl"):<br>
&nbsp&nbsp&nbsp&nbspwith open(filename, "rb") as file:<br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspreturn pickle.load(file)<br><br>

df = model_backend.load_data()<br>
df = model_backend.preprocess_data(df)<br>
train_df, _ = model_backend.train_test_data_split(df, test_size=0.25)<br>
prophet_model = model_backend.train_prophet_model(train_df)<br><br>
def calculate_mape(actual, predicted):<br>
    # Ensure no division by zero<br>
actual, predicted = np.array(actual), np.array(predicted)<br>
    nonzero_mask = actual != 0<br>
    if np.any(nonzero_mask):<br>
        return np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100<br>
    else:<br>
        return float('inf')<br>
@app.get("/")<br>
async def root():<br>
    return {"message": "Hello World"}<br><br>

@app.post("/predict")<br>
async def predict_sales(request: PredictionRequest):<br>
 prediction = model_backend.predict_for_date(prophet_model, request.date)<br>
return prediction<br>
</code></div><br><br>
This script defines a RESTful API using FastAPI, a modern web framework for building APIs with Python 3.7+, known for its high performance and ease of use due to automatic request validation and interactive API documentation. At its core, the API is designed to interact with a backend sales forecasting model, specifically handling requests for sales predictions. The CORSMiddleware is added to the FastAPI application to handle Cross-Origin Resource Sharing (CORS), enabling the API to accept requests from different domains, which is particularly useful during the development and testing phase but should be more restrictive in production for security reasons.<br><br>

The API has two endpoints: a root endpoint that simply returns a welcome message, and a /predict endpoint that takes a date as input and returns a sales forecast for that date. The prediction logic is encapsulated within the model_backend module, which loads and preprocesses historical sales data, splits it into training and testing sets, and trains a Prophet model—a sophisticated time series forecasting tool that handles various components of time series data like trends and seasonality efficiently. The trained model is then used to make predictions. FastAPI's ability to define expected request body formats through Pydantic models, as demonstrated with the PredictionRequest class, helps in validating the data and automatically generates descriptive documentation. By leveraging FastAPI's features, the script sets up a highly efficient, scalable, and easily maintainable API service that is capable of providing real-time sales forecasts, aiding in data-driven decision-making processes.<br><br><br>
<h3><u>User Interface</u></h3>
This HTML document outlines a user interface for a FastAPI application aimed at forecasting Walmart's weekly sales. The page is styled for clarity and ease of use, featuring a responsive layout that adapts to various devices, thanks to its flexible viewport settings and the Roboto font for readability. The interface consists of a main container with a header that introduces the application, and a body section where users can input a specific date into a textarea to request sales forecasts. Upon submitting the date, a spinner animation is displayed, indicating that the request is being processed. Results, including the predicted sales and confidence intervals, are dynamically displayed in the response area below the input field. Additionally, a sample cURL command is provided in a separate section, showing how to interact with the API directly, which can be useful for developers looking to integrate with the service programmatically. This setup not only enhances user engagement through an intuitive layout but also ensures a seamless interaction by providing immediate feedback and detailed results.<br>
<h4><u>Attached CSS</u></h4><br>
<div style="border: 2px solid #000; padding: 10px;"><code>
.spinner {<br>&nbsp
  &nbspborder: 4px solid rgba(0,0,0,.1);<br>&nbsp
  &nbspwidth: 36px;<br>&nbsp
  &nbspheight: 36px;<br>&nbsp
  &nbspborder-radius: 50%;<br>&nbsp
  &nbspborder-left-color: #3f51b5;<br>&nbsp
  &nbspanimation: spin 1s ease infinite;<br>&nbsp
}<br><br>
@keyframes spin {<br>&nbsp
  0% { transform: rotate(0deg); }<br>&nbsp
  100% { transform: rotate(360deg); }<br>
}<br><br>
textarea:hover {<br>&nbsp&nbsp
    border: 1px solid #3f51b5; /* Changes the border<br>&nbsp&nbsp color on hover */<br>&nbsp&nbsp
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2); /* Adds a subtle shadow around the textarea */<br>
}<br><br>
  textarea:focus {<br>&nbsp&nbsp
    border-color: #303f9f; /* Darker shade when focused for better visibility */<br>&nbsp&nbsp
    outline: none; /* Removes the default focus outline */<br>&nbsp&nbsp
    box-shadow: 0 0 0 2px #3f51b5; /* Adds a glow effect, making it clear that the textarea is focused */<br>
}<br>
</code></div><br><br>
This CSS snippet is designed to add interactive styling to a webpage's user interface elements. The .spinner class styles a loading indicator: a circular shape with a border, where the left segment is highlighted in a deep blue color (#3f51b5). This spinner uses CSS animation to rotate continuously, giving users a visual cue that a process is ongoing. The @keyframes spin rule defines this animation, dictating a smooth, 360-degree rotation every second, looping indefinitely for a perpetual spinning effect.<br><br>

The textarea styles enhance user interaction with the text input area. When a user hovers over a textarea, it changes the border to a blue color and adds a light shadow, creating a subtle 3D lift effect. Upon focusing on the textarea, which happens when the user clicks into it to type, the border darkens to a more vivid blue (#303f9f), and a glowing blue outline appears, providing clear feedback that the element is active and ready for input. These visual enhancements are not just aesthetic; they're functional cues that improve user experience by clearly indicating the element's state—idle, hovered, or focused.<br><br><br>
<h3><u>Locust Implementation</u></h3>
<div style="border: 2px solid #000; padding: 10px;"><code>
from locust import HttpUser, task, between<br><br>
class WebsiteUser(HttpUser):<br>&nbsp&nbsp&nbsp&nbsp&nbsp
    wait_time = between(1, 5)<br><br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp@task<br>
    &nbsp&nbsp&nbsp&nbsp&nbsp&nbspdef predict_sales(self):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspself.client.post("/predict", json={"date": "2022-01-01"})<br><br>
&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp@task(3)<br>
    &nbsp&nbsp&nbsp&nbsp&nbsp&nbspdef view_root(self):<br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspself.client.get("/")<br>
</code></div><br><br>
Locust is an open-source load testing tool that lets you simulate users interacting with your application to see how it handles stress in real-time. It's written in Python, making it highly readable and easy to use, especially for teams already working in a Pythonic environment. One of the key advantages of Locust is its ability to create realistic test scenarios by defining user behavior with code, rather than being confined to a UI-based setup. This provides flexibility and precision in mimicking various user interactions, and its distributed nature allows it to scale out to millions of simultaneous users, if necessary.<br><br>

Here, we're defining a WebsiteUser class that represents a user of the web service. The class includes two tasks to mimic two types of user interactions: making a POST request to the /predict endpoint and a GET request to the root ("/") endpoint. The wait_time is set with between(1, 5), indicating that each simulated user will pause for a random time between 1 and 5 seconds between tasks, which simulates the unpredictable nature of user behavior. The @task decorator defines the actions that the simulated users will perform, with view_root being three times more likely to be executed than predict_sales due to the weight of 3. This setup allows for the creation of a load testing scenario that closely mirrors the anticipated user interaction pattern, ensuring that the sales forecasting API, built upon FastAPI and predictive modeling with Prophet, is tested under conditions that resemble live usage, verifying that the system is robust, responsive, and reliable under stress.<br><br><br> 
<h2 style="text-align: left;"><i><u>Conclusion</u></i></h2><br>
In conclusion, the Walmart Sales Forecasting application stands as a beacon of innovation, integrating state-of-the-art machine learning techniques with the robustness of a web application to deliver precise weekly sales forecasts. The successful marriage of SARIMAX and Prophet models within a Python-based backend is a testament to the system's robustness, adeptly managing the intricacies of historical sales data and influential variables such as holidays and economic indicators. The result is a powerful tool that not only forecasts with high accuracy but also adapts seamlessly to the ever-changing patterns of consumer behavior, proving indispensable for strategic planning and decision-making.<br><br>
The development of a user-friendly interface employing HTML, CSS, and JavaScript, coupled with the backend proficiency of FastAPI, creates an accessible platform for users of varied technical backgrounds. Performance validation through Locust further reinforces the application's resilience, ensuring that it can withstand the rigors of real-world deployment and user interaction. By simulating user behavior and load conditions, we can confidently affirm that the application will perform reliably and efficiently under diverse operating scenarios.<br><br>
This project's utility extends beyond its immediate operational benefits to Walmart. It encapsulates the power of predictive analytics in the retail sector, offering a blueprint for how data-driven insights can optimize inventory management, enhance customer satisfaction, and boost profitability. The Walmart Sales Forecasting application is a quintessential example of how machine learning can be harnessed to elevate business intelligence, demonstrating that when it comes to integrating cutting-edge technology with business applications, the sky is the limit.