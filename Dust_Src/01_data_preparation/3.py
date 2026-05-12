import pandas as pd
df = pd.read_csv('dataset_with_new_labels_core.csv')
# print(df.columns)
# Select the first 7 unique days (timestamps)
first_7_days = sorted(df['timestamp'].unique())[:7]
df_7days = df[df['timestamp'].isin(first_7_days)]
df_7days.to_csv('7day.csv', index=False)