# CNN-LSTM-CTC text recognition

I realize three different models for text recognition, and all of them consist of CTC loss layer to realize no segmentation for text images.

### Disclaimer

I refer to the official mxnet warpctc example [here](https://github.com/dmlc/mxnet/tree/master/example/warpctc).

### Getting started
* Build MXNet with Baidu Warp CTC, and please following this instructions [here](https://github.com/dmlc/mxnet/tree/master/example/warpctc).

When I use this official instructions to add Baidu Warp CTC to Mxnet, there are some errors because the latest version of Baidu Warp CTC has complicts with mxnet. Recently, I see someone has already solved this problem and updated the official mxnet warpctc example. However, if you still have problem, please refer to this issue [here](https://github.com/dmlc/mxnet/pull/3853).

### Generating data

Run `generate_data.py` in `generate_data`. When generating training and test data, please remember to change output path and number in `generate_data.py` (I will update a more friendly way to generate training and test data when I have free time).

### Train the model

I realize three different models for text recognition, you can check them in `symbol`:

1. LSTM + CTC;
2. Bidirection LSTM + CTC;
3. CNN (a modified model similiar to VGG) + Bidirection LSTM + CTC. Disclaimer: This CNN + LSTM + CTC model is a re-implementation of original CRNN which is based on torch. The official repository is available [here](https://github.com/bgshih/crnn). The arxiv paper is available [here](https://arxiv.org/pdf/1507.05717v1.pdf).

* Start training:

LSTM + CTC:

```
python train_lstm.py
```

Bidirection LSTM + CTC:

```
python train_bi_lstm.py
```

CNN + Bidirection LSTM + CTC:

```
python train_crnn.py
```
### Prediction

You can do the prediction using your trained model. I only write the predictors for model 1 and model 3, but it is very easy to write the predictor for model 2 when referring to the examples.

Plesae run:
```
python lstm_predictor.py
```
or
```
python crnn_predictor.py
```



