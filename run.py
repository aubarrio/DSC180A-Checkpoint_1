import argparse
import sys
import json

from src.data.etl import complete

def main():
    if targets:
        assert len(targets) == 2, "Must provide file path for files example: data/raw/twitch/  ; along with source of data example: twitch"
        complete(targets[0], targets[1])
    else:
        with open('config/params.json') as fh:
            data_cfg = json.load(fh)
        complete(**data_cfg)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main()

    #examples: python run.py data/raw/facebook facebook
    #          python run.py data/raw/twitch twitch
    #          python run.py data/raw/cora cora

    #The following command would run the default cora dataset
    #          python run.py
