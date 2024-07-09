# TGML-AUT
Temporal Graph Networks (TGNs) generalize Message Passing Neural Networks (MPNNs) to temporal graphs. They do so by introducing a node memory which represents the state of the node at a given time, acting as a compressed representation of the node’s past interactions. Every time two nodes are involved in an interaction, they send messages to each other which are then used to update their memories. When computing the embedding of a node, an additional graph aggregation is performed over the temporal neighbors of the node, using both the original node features and memory at that point in time. 


## Dataset tgbn-trade :
Temporal Graph: The tgbn-trade dataset is the international agriculture trading network between nations of the United Nations (UN) from 1986 to 2016. Each node is a nation and an edge represents the sum trade value of all agriculture products from one nation to another one. As the data is reported annually, the time granularity of the dataset is yearly.

Prediction task: The considered task for this dataset is to predict the proportion of agriculture trade values from one nation to other nations during the next year.

References
[1] G. K. MacDonald, K. A. Brauman, S. Sun, K. M. Carlson, E. S. Cassidy, J. S. Gerber, and P. C. West. Rethinking agricultural trade relationships in an era of globalization. BioScience,65(3):275–289, 2015.

License: MIT license
