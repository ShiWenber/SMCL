# SMCL

use the following command to create a conda environment from environment.yml file.

```bash
conda env create -n <env_name> -f environment.yml
```

## A Simple Self-Supervised Graph Representation via Online Predict

Self-supervised graph representation learning is a key technique for graph structured data processing, especially for Web-generated graph that do not have qualified labelling information.
## Dependencies

```python
pip install -r requirements.txt
```

## Usage

You can use the following command, and the parameters are given

For node classification task:
```python
python main.py --dataset Cora
```

The `--dataset` argument should be one of [Cora, CiteSeer, PubMed, Com, Photo, Phy, CS, WikiCS].
