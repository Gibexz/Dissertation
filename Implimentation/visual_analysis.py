import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv('./Datasets/Extracted_Data/ycrit_weekly_w_mon_copy.csv')
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
df.sort_index(inplace=True)

# Generate Visual Trend Plot
plt.figure(figsize=(12, 6))
plt.plot(df['ycrit'], color='blue', label='Weekly Velocity (Ycrit)')
plt.title('Weekly High/Critical Vulnerability Velocity (2019-2023)')
plt.xlabel('Week')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('Visualisations/visual_trend_analysis.png')