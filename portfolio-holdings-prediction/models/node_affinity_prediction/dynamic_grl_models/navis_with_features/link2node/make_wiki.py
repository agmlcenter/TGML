import pandas as pd
import csv

DATA_PATH = "tgbl-wiki_edgelist_v2.csv"
DIR = "wiki"

# Read the CSV file
data = csv.reader(open(f"{DIR}/{DATA_PATH}", "r"))

# Leave only the first three columns without pandas
data = [row[:3] for row in data]

# Save new dataframe
with open(f"{DIR}/tgbn-wiki_edgelist_v3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)


# Load new csv 
data = pd.read_csv(f"{DIR}/tgbn-wiki_edgelist_v3.csv")

# Add additional column of 'weight' where each row has a value of 1
data['weight'] = 1

# Reorder columns to ts,source,target,weight
data = data[['timestamp', 'user_id', 'item_id', 'weight']]

# Save the dataframe to a csv file
data.to_csv(f"{DIR}/tgbn-wiki_edgelist.csv", index=False)

# Split rows into groups by timestamps where each shard i has timestamp in range [i*86400, (i+1)*86400) 
shards = []
for i in range(30):
    shard = data[(data['timestamp'] >= i * 86400) & (data['timestamp'] < (i + 1) * 86400)]
    shards.append(shard)


# Row in a shard i has timestamp (i + 1) * 86400
for i, shard in enumerate(shards):
    #shard['timestamp'] = (i + 1) * 86400
    shard.loc[:, 'timestamp'] = i  * 86400


# Aggregate each shard by summing the weights of each user_id,item_id tuple
aggregated_shards = []
for shard in shards:
    shard = shard.groupby(['user_id', 'item_id'], as_index=False).agg({'weight': 'sum','timestamp': 'first'})
    aggregated_shards.append(shard)

# concat
aggregated = pd.concat(aggregated_shards)

# Normalise weights for each user_id, s.t. for each user at each timestamp, summing weights for all of its item_ids at the same timestamp will be 1
aggregated['weight'] = aggregated.groupby(['user_id', 'timestamp'])['weight'].transform(lambda x: x / x.sum())

# Remove rows with timestamp 0
aggregated = aggregated[aggregated['timestamp'] != 0]

# Reorder columns to timestamp,user_id,item_id,weight
aggregated = aggregated[['timestamp', 'user_id', 'item_id', 'weight']]

# Make sure that timestamp type is int
aggregated['timestamp'] = aggregated['timestamp'].astype(int)

# Save labels
aggregated.to_csv(f"{DIR}/tgbn-wiki_node_labels.csv", index=False)

# print number of different item_ids
print(aggregated['item_id'].nunique())
