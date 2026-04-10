# Prepare the Dataset
## Download Dataset
Download [IoTID20](https://sites.google.com/view/iot-network-intrusion-dataset/home) <br>
Download [CiC-BoTIoT](https://espace.library.uq.edu.au/view/UQ:c80fccd)

## Split in Benign and Anomaly

Run `python3 split_whole_dataset.py` for each dataset. To split into benign and malicious

## Split in á 1000 Sample splits

run `python3 split_in_splits.py` for each subset of benign and malicious.

## Split in á 1000 Samples for Training and Server eval

Run ` python3 split_in_splits_sorted.py` for pure benign subset of each dataset.

`_glo` files are used in server evaluation and the resulting splits used for training.
