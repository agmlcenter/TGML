# Instructions for setting up new tgbn datasets

1. Go inside tgb package directory
2. Go into datasets directory and create a new directory for the dataset:
`cd datasets`
`mkdir tgbn_<dataset_name>`
3. Copy raw data of tgbl-<dataset_name> inside tgbn_<dataset_name> and process it using the provided appropriate make_<dataset_name>.py script:
`cp ../tgbl-<dataset_name>/<data_file>.csv tgbn_<dataset_name>/`
`python make_<dataset_name>.py`
4. Inside tgb/utils/info.py, add the following key-value pair to DATA_NUM_CLASSES: `"tgbn-<dataset_name>": <num_classes>`
The number of classes is printed at the end of the make_<dataset_name>.py script.
5. Add the following key-value pair to DATA_EVAL_METRIC_DICT:
`"tgbn-<dataset_name>": ndcg`.
6. In tgb/nodeproppred/dataset.py add the following lines after the elif statement in line 237 for ['wikipedia']:
````elif self.name == "tgbn-<dataset_name>":
                df, edge_feat, node_ids, labels_dict = load_edgelist_datetime(
                    self.meta_dict["fname"], label_size=self._num_classes
                )
````
and for ['flights','uslegis', 'unvote']:
````elif self.name == "tgbn-<dataset_name>":
                df, edge_feat, node_ids = load_edgelist_trade(
                    self.meta_dict["fname"], label_size=self._num_classes
                )
7. Additionally, for ['flights','uslegis', 'unvote'] add `or self.name == "tgbn-flights"`
In the `if self.name == "tgbn-trade":` condition located right after that.