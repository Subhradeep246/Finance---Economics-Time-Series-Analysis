import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('finance_economics_dataset.csv')

df['Date'] = pd.to_datetime(df['Date'])

df['Year'] = df['Date'].dt.year

annual_data = df.groupby('Year').agg({
    'Close Price': 'mean',
    'GDP Growth (%)': 'mean',
    'Inflation Rate (%)': 'mean',
    'Unemployment Rate (%)': 'mean'
}).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Economic Indicators Over Time', fontsize=16, fontweight='bold')

axes[0, 0].bar(annual_data['Year'], annual_data['Close Price'], color='steelblue', alpha=0.7)
axes[0, 0].set_title('Average Stock Close Price (by Year)', fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].grid(axis='y', alpha=0.3)

axes[0, 1].bar(annual_data['Year'], annual_data['GDP Growth (%)'], color='coral', alpha=0.7)
axes[0, 1].set_title('GDP Growth (%)', fontweight='bold')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Percentage')
axes[0, 1].grid(axis='y', alpha=0.3)

axes[1, 0].bar(annual_data['Year'], annual_data['Inflation Rate (%)'], color='mediumseagreen', alpha=0.7)
axes[1, 0].set_title('Inflation Rate (%)', fontweight='bold')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Percentage')
axes[1, 0].grid(axis='y', alpha=0.3)

axes[1, 1].bar(annual_data['Year'], annual_data['Unemployment Rate (%)'], color='goldenrod', alpha=0.7)
axes[1, 1].set_title('Unemployment Rate (%)', fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Percentage')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig('economic_indicators.png', dpi=300, bbox_inches='tight')
print("Chart saved as 'economic_indicators.png'")
plt.show()