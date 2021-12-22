# DegreEmbed: incorporating entity embedding into logic rule learning for knowledge graph reasoning

This is the PyTorch implementation of the model DegreEmbed, proposed in the paper [DegreEmbed: incorporating entity embedding into logic rule learning for knowledge graph reasoning](https://arxiv.org/abs/2112.09933).

Yuliang Wei, Haotian Li, Yao Wang, Guodong Xin, Hongri Liu

## Environment setup

- Python >= 3.7
- PyTorch >= 1.8.0
- NumPy
- tqdm

It is recommended to create a conda virtual environment using the `requirements.yaml` by the following command:

```shell
conda create --name DegreEmbed --file requirements.yaml
```

## Quick start

We provide a demo for training and evaluating our model on the `Family` dataset. All datasets can be found in the `datasets` folder.

### Training

Run the following command in you shell **under the root directory** of this repository to train a model.

```shell
python model/main.py --dataset=family
```

You may check the configuration file `model/configure.py` for more possible hyperparameter combinations. There is also a jupyter notebook for training this model in an interactive way locating at `model/train.ipynb`.

When the training process finishes, there are extra files created by the script that are stored under the directory `saved/family`, e.g., `option.txt` contains hyperparameters in this experiment and `prediction.txt` is the prediction results on test data for computing the metrics MRR (Mean Reciprocal Rank) and Hit@k.

### Evaluation

1. MRR & Hit@k

There is a separate script `eval/evaluate.py` to compute the MRR and Hit@k under the filtered protocol proposed in [TransE](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), and you will see the evaluation result in your CLI.

```shell
python eval/evaluate.py --dataset=family --include_reverse --top_k=10 --filter
```

You may also find another shell script `eval/evaluate.sh` helpful to evaluate the model using different configurations in `eval/evaluate.py`.

2. Mined rules and saturations

You can run the notebook `model/train.ipynb` to train a DegreEmbed model as well as generate logic rules on the certain dataset.

The computation of saturations can be accessed in two notebooks, `datasets/graph_assessment.ipynb` and `datasets/graph_assessment-multihop.ipynb`, the former for saturations of rules with fixed length of two while the latter one allows varied lengths no longer than $L$.

For more details, please check the jupyter notebooks mentioned above.

## Citation

If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2112.09933):

```plaintext
@misc{wei2021degreembed,
      title={DegreEmbed: incorporating entity embedding into logic rule learning for knowledge graph reasoning}, 
      author={Yuliang Wei and Haotian Li and Yao Wang and Guodong Xin and Hongri Liu},
      year={2021},
      eprint={2112.09933},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
