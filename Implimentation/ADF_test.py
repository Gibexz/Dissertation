import pandas as pd
import matplotlib.pyplot as plt


# Load and prepare the dataset
df = pd.read_csv('./Datasets/Extracted_Data/ycrit_weekly_w_mon_copy.csv')
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
df.sort_index(inplace=True)

from statsmodels.tsa.stattools import adfuller

# Execute ADF Test
adf_result = adfuller(df['ycrit'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")