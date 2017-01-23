# pylint:skip-file
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states", "forward_state", "backward_state",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


def crnn(num_lstm_layer, seq_len, num_hidden, num_classes, num_label, dropout=0.):
    
    last_states = []
    forward_param = []
    backward_param = []
    for i in range(num_lstm_layer*2):
      last_states.append(LSTMState(c = mx.sym.Variable("l%d_init_c" % i), h = mx.sym.Variable("l%d_init_h" % i)))
      if i % 2 == 0:
        forward_param.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                  i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                  h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                  h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
      else:
        backward_param.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                  i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                  h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                  h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))

    # input
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    #CNN model- similar to VGG
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    # conv1_2 = mx.symbol.Convolution(
    #     data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    # relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    # conv2_2 = mx.symbol.Convolution(
    #     data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    # relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    batchnorm1 = mx.symbol.BatchNorm(data= conv3_1, name="batchnorm1")
    relu3_1 = mx.symbol.Activation(data=batchnorm1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    # conv3_3 = mx.symbol.Convolution(
    #     data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    # relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 1), stride=(2, 1), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    batchnorm2 = mx.symbol.BatchNorm(data= conv4_1, name="batchnorm2")
    relu4_1 = mx.symbol.Activation(data=batchnorm2, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=batchnorm1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    # conv4_3 = mx.symbol.Convolution(
    #     data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    # relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=batchnorm2, pool_type="max", kernel=(2, 2), stride=(2, 1), pad=(0, 1), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(2, 2), pad=(0, 0), num_filter=512, name="conv5_1")
    batchnorm3 = mx.symbol.BatchNorm(data= conv5_1, name="batchnorm3")
    # relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    # conv5_2 = mx.symbol.Convolution(
    #     data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    # relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    # conv5_3 = mx.symbol.Convolution(
    #     data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    # relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    # pool5 = mx.symbol.Pooling(
    #     data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),
    #     pad=(1,1), name="pool5")
    if dropout > 0.:
        batchnorm3 = mx.sym.Dropout(data=batchnorm3, p=dropout)
    # arg_shape, output_shape, aux_shape = batchnorm3.infer_shape(data=(32,1,32,100))
    # print(output_shape)
    cnn_out = mx.sym.transpose(data=batchnorm3, axes=(0,3,1,2), name="cnn_out")
    # arg_shape, output_shape, aux_shape = cnn_out.infer_shape(data=(32,1,32,100))
    # print(output_shape)
    flatten_out = mx.sym.Flatten(data=cnn_out, name="flatten_out")
    # arg_shape, output_shape, aux_shape = flatten_out.infer_shape(data=(32,1,32,100))
    # print(output_shape)
    wordvec = mx.sym.SliceChannel(data=flatten_out, num_outputs=seq_len, squeeze_axis=1)
    
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        for i in range(num_lstm_layer): 
          next_state = lstm(num_hidden, indata=hidden,
                            prev_state=last_states[2*i],
                            param=forward_param[i],
                            seqidx=seqidx, layeridx=0, dropout=dropout)
          hidden = next_state.h
          last_states[2*i] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = wordvec[k]
        for i in range(num_lstm_layer):
          next_state = lstm(num_hidden, indata=hidden,
                            prev_state=last_states[2*i + 1],
                            param=backward_param[i],
                            seqidx=k, layeridx=1,dropout=dropout)
          hidden = next_state.h
          last_states[2*i + 1] = next_state
        backward_hidden.insert(0, hidden)

        
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_classes)

    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)

    return sm

