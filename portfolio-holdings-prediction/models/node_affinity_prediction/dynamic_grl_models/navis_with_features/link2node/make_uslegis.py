import pandas as pd
import csv

DATA_PATH = "USLegis.csv"
DIR = "."

# Load new csv 
data = pd.read_csv(f"{DIR}/USLegis.csv")

# Change column name 'w' to 'weight'
data = data.rename(columns={'w': 'weight'})

# Reorder columns to ts,source,target,weight
data = data[['timestamp', 'source', 'destination', 'weight']]

# Save the dataframe to a csv file
data.to_csv(f"{DIR}/tgbn-uslegis_edgelist.csv", index=False)

# Get unique timestamps
unique_timestamps = data['timestamp'].unique()
unique_timestamps.sort()

# Create shards based on unique timestamps
shards = []
for ts in unique_timestamps:
    shard = data[data['timestamp'] == ts]
    shards.append(shard)

# Aggregate each shard by summing the weights of each user_id,item_id tuple
aggregated_shards = []
for shard in shards:
    shard = shard.groupby(['source', 'destination'], as_index=False).agg({'weight': 'sum','timestamp': 'first'})
    aggregated_shards.append(shard)

# concat
aggregated = pd.concat(aggregated_shards)

# Normalise weights for each source, s.t. for each source at each timestamp, summing weights for all of its destinations at the same timestamp will be 1
aggregated['weight'] = aggregated.groupby(['source', 'timestamp'])['weight'].transform(lambda x: x / x.sum())

# Remove rows with timestamp 0
aggregated = aggregated[aggregated['timestamp'] != 0]

# Reorder columns to timestamp,source,destination,weight
aggregated = aggregated[['timestamp', 'source', 'destination', 'weight']]

# Make sure that timestamp type is int
aggregated['timestamp'] = aggregated['timestamp'].astype(int)

# Save labels
aggregated.to_csv(f"{DIR}/tgbn-uslegis_node_labels.csv", index=False)

# print number of different labels
print(225)
