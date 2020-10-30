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


# python run.py data/raw/twitch/ENGB/musae_ENGB_features.json data/raw/twitch/ENGB/musae_ENGB_edges.csv

#make sure to change to snap instead of cora
