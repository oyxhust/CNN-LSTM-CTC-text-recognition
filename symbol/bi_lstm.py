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


def bi_lstm_unroll(num_lstm_layer, seq_len, num_hidden, num_classes, num_label, dropout=0.):
    
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

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    
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


def bi_lstm_inference_symbol(input_size, seq_len,
                             num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = [LSTMState(c = mx.sym.Variable("l0_init_c"), h = mx.sym.Variable("l0_init_h")), 
                   LSTMState(c = mx.sym.Variable("l1_init_c"), h = mx.sym.Variable("l1_init_h"))]
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    data = mx.sym.Variable("data")
    embed = mx.sym.Embedding(data=data, input_dim=input_size,
                             weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    forward_hidden = []
    for seqidx in range(seq_len):
        next_state = lstm(num_hidden, indata=wordvec[seqidx],
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0, dropout=0.0)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        next_state = lstm(num_hidden, indata=wordvec[k],
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1, dropout=0.0)
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)
        
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
                               weight=cls_weight, bias=cls_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)

