## Usage

You can use the following command, and the parameters are given

For graph classification task:
```python
python main.py --dataset ENZYMES
```

The `--dataset` argument should be one of [ENZYMES, PTC-MR, NCI1, NCI109, PROTEIN, DD, MUTAG, IMDB-B, IMDB-M, COLLAB, RDT-M5].

查看显卡占用

```bash
nvidia-smi
```

查看当前后台任务，根据string搜索

```bash
ps -ef | grep "string"
```