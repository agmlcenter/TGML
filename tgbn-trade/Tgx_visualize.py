# https://github.com/ComplexData-MILA/TGX/blob/master/docs/tutorials/data_viz_stats.ipynb

import tgx
from tgx.utils.plotting_utils import plot_for_snapshots

dataset = tgx.tgb_data("tgbn-trade")
ctdg = tgx.Graph(dataset)

dataset = tgx.builtin.uci()
ctdg = tgx.Graph(dataset)
time_scale = "weekly" #"minutely", "hourly", "daily", "monthly", "yearly", "biyearly"
dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)


tgx.degree_over_time(dtdg, network_name="uci") # Plot the average node degree over time
# tgx.nodes_over_time(dtdg, network_name="uci") # Plot the number of active nodes per snapshot
# tgx.edges_over_time(dtdg, network_name="uci") # Plot the number of edges per snapshot
# tgx.nodes_and_edges_over_time(dtdg, network_name="uci") #Plot the number of active nodes and edges in the same figure
# tgx.connected_components_per_ts(dtdg, network_name="uci") # Plot the number of connected components per timestamp.
# tgx.degree_density(dtdg, network_name="uci") # Plot the density map of node degrees per time window
# tgx.TEA(dtdg, network_name="uci") # Plot Temporal Edge Appearance (TEA) (from Poursafaei et al. 2022)
# tgx.TET(dtdg, network_name="uci") # Plot Temporal Edge Traffic (TET) from (Poursafaei et al. 2022)

# tgx.get_reoccurrence # Calculate the recurrence index	float
# tgx.get_surprise # Calculate the surprise index	float
# tgx.get_novelty # Calculate the novelty index	float
# tgx.get_avg_node_activity # Calculate the average node activity	float
# tgx.size_connected_components # Calculate the sizes of connected components	List[List[float]]
# tgx.get_avg_node_engagement # Calculate the average node engagement	List[float]
