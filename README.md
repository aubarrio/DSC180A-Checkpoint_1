# DSC180A-Checkpoint_1

How to run the run.py file.
This file reads in a dataset and runs a 2 Layer GCN on the dataset.

## 3 Different Datasets
* the cora dataset
* a twitch dataset found on SNAP
* a facebook dataset also found on SNAP

## In order to run run.py through termal you must use: 
* `python run.py <file path> source`

If you were to run this on the twitch dataset then you would run the command
* `python run.py data/raw/twitch twitch`

Examples for the other two datasets are as follows
* `python run.py data/raw/cora cora`
* `python run.py data/raw/facebook facebook`

Simply running the command
              `python run.py`
would default to using the cora dataset gathering its params from the config/params.json file


### Responsibilities
* Austin Le - Added some functionaliaty to ingestion pipeline, Crafted Introduction
* Aurelio Barrios
