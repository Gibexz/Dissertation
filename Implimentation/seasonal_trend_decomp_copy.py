import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load and prepare the dataset
df = pd.read_csv('./Datasets/Extracted_Data/ycrit_weekly_w_mon_copy.csv')
df['week'] = pd.to_datetime(df['week'])
df.set_index('week', inplace=True)
df.sort_index(inplace=True)

# STL Decomposition
stl = STL(df['ycrit'], period=52)
res = stl.fit()

fig = res.plot()

# 🔥 Apply zoom to ALL subplots
for ax in fig.axes:
    ax.set_xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2023-12-31'))

fig.suptitle('STL Decomposition of Weekly Vulnerability Velocity (2022–2023)', fontsize=14)

plt.tight_layout()

plt.savefig('Visualisations/stl_decomposition_jan-2022-to-dec-2023.png', dpi=300, bbox_inches='tight')
plt.show()