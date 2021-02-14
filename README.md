# invase-gnn

Extension of INVASE to GNNs (work in progress). 

This directory contains implementations of INVASE framework for 
the following applications on graph input data.

-   Instance-wise feature and node selection
-   Prediction with instance-wise feature and node selection

To run the pipeline for training and evaluation on INVASE-GNN framwork on the MUTAG dataset,
simply run python main.py. Experiment and model hyperparameters can be adjusted using command line flags. 

## Example Usage

```shell
$ python3 main.py --model-type INVASE --task mutag \
                --node-lamda 0.01 --fea-lamda 0.01 --l2 0.0 --dropout 0.0 \
                --actor-h-dim 64 --critic-h-dim 64 --n-layer 3 \
                --batch-size 256 --epochs 300 --lr 0.01 \
                --run-id 1 
```

## TODO
- Comparison to other methods
- Synthetic data experiments
- Node classification feature selection

## Acknowledgements

Original paper:
```
Jinsung Yoon, James Jordon, Mihaela van der Schaar, 
"INVASE: Instance-wise Variable Selection using Neural Networks," 
International Conference on Learning Representations (ICLR), 2019.
(https://openreview.net/forum?id=BJg_roAcK7)
```

This code was built on top of the [INVASE code repository](https://github.com/jsyoon0823/INVASE), and some inspiration was taken from [invase-pytorch](https://github.com/mertyg/invase-pytorch).


