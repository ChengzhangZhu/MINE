# MINE
This code is a Keras implementation (only for tensorflow backend) of MINE: Mutual Information Neural Estimation (https://arxiv.org/pdf/1801.04062.pdf)

Thank *mzgubic* for providing the [Tensorflow implementation](https://github.com/mzgubic/MINE/blob/master/MINE_in_TF.ipynb)

## How to use

*Option 1*: declare the dimension of each input; this build a 3-layer fully-connected network as the statistics network 

```python
from mine import MINE
mine = MINE(x_dim=100, y_dim=200)
fit_loss_history, mutual_info = mine.fit(x, y)
```

*Option 2*： predefine a statistics network by Keras Model 

```python
from mine import MINE
network = Model(inputs=[x_input, y_input], outputs=outputs) # a Keras model
mine = MINE(network=network)
fit_loss_history, mutual_info = mine.fit(x, y)
```

Several parameters can be set in the fitting function:

`fit(x, y, epochs=50, batch_size=100, verbose=1)`

paras: 
+ `x`: list or np.array, the input of samples drawn from the first distribution
+ `y`: np.array, the input of samples drawn from the second distribution
+ `epochs`: int, the number of training epochs (default: 50)
+ `batch_size`: int, the training batch size (default: 100)
+ `verbose`: 0 or 1, whether print training process

returns:
 + `fit_loss_history`: list, the history of fitting loss values
 + `mutual_info`: float, the estimated mutual information
 
 A demo has been attached in *demo.ipynb*. Enjoy :)