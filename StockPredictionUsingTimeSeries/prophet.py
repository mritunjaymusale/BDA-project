import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv('../data/GOOGL.csv')
df.columns = ['ds','Open','High','Low','Close','Adj Close','y']
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=100)
forecast = model.predict(future)
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
plt.show()