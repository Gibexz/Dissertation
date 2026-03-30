import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv('./Datasets/Extracted_Data/ycrit_weekly_w_mon_copy.csv')
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
df.sort_index(inplace=True)


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Generate ACF and PACF Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['ycrit'], ax=ax1, lags=52)
plot_pacf(df['ycrit'], ax=ax2, lags=52, method='ywm')
ax1.set_title('Autocorrelation (ACF)')
ax2.set_title('Partial Autocorrelation (PACF)')
plt.tight_layout()
plt.savefig('Visualisations/acf_pacf_analysis.png')
plt.show()