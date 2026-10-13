# Instructions for training and testing NAVIS

## Setup
1. Install all the requirements for running TGB as listed in the [TGB docs](https://docs.tgb.complexdatalab.com/).
2. Install the `pytorchltr` package by following the instructions in the [pytorchltr docs](https://pytorchltr.readthedocs.io/en/latest/).
3. Inside "pytorchltr/loss/pairwise_lambda.py" Add `exponent = exponent.clamp(0,30)` after line 115 to avoid instability issues.


## Running NAVIS
Run NAVIS by executing the following command:
`navis.py`

You can modify the hyperparameters in the `navis.py` file, listed under "hyperparameters".
