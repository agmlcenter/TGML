import pandas as pd

# Read the original CSV
df = pd.read_csv("TEB_static_per_node.csv")

# Select the first 5 rows
df_head = df.head(5)

# Save to a new CSV file
df_head.to_csv("STATIC.csv", index=False)
