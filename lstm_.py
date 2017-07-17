import numpy as np

import torch as tc

from collections import namedtuple
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias",
                                     "aux2h_weight", "aux2h_bias"])


URNNParam = namedtuple("URNNParam", ["W1_a", "W1_b",
                                     "W2_a", "W2_b",
                                     "W3_a", "W3_b"])

ULSTMParam = namedtuple("ULSTMParam", ["W1_a", "W1_b",
                                      "W2_a", "W2_b",
                                      "W3_a", "W3_b",
                                      "W4_a", "W4_b",
                                      "W5", "W6",
                                      "Bias1_a", "Bias1_b",
                                      "Bias2_a", "Bias2_b"])

def orth(M):
    size = M.size(0)
    M = list(M.split(1))
    norm_b = M[0]
    w_b = tc.mm(norm_b, norm_b.t())
    for k in range(1, size):
        norm_a = tc.mm(M[k], norm_b.t()) / w_b
        residual = tc.mm(norm_a, norm_b)
        M[k] -= residual
        norm_b = tc.cat([norm_b, M[k]], dim=0)
        w_b = tc.cat([w_b, tc.mm(M[k], M[k].t())], dim=1)
    return norm_b / tc.norm(norm_b, p=2, dim=1, keepdim=True)

z = tc.randn(10, 10)
z = orth(z)
print z
print tc.mm(z, z.t())

def initUMatrix(prefix, size):
    p1 = tc.Tensor(size, size)
    p2 = tc.Tensor(size, size)
    p3 = tc.Tensor(1, size)
    U_a, U_b = Unitry(p1, p2, p3, size)
    return U_a, U_b

def initU(M, size ,eye):
    #M = mx.sym.broadcast_add(M*eye, M*(1-eye))
    #M = M * eye + M * (1 - eye)
    return orth(orth(M + M.t(), size), size)
    '''
    inversable_matrix = mx.sym.broadcast_add(mx.sym.abs(M), 1e-20 + mx.sym.sum(mx.sym.abs(M), axis=0, keepdims=True))*eye + M*(1-eye)  
    return orth(inversable_matrix, dim0_size)
    '''

def initE(vec, size):
    M = vec.repeat(size, 1)
    eye = tc.eye(size)
    return M * eye, eye

def Unitry(p1, p2, p3):

    size = p3.size(-1)     # (1, size)
    E, eye = initE(p3, size)

    Q = initU(p1, size, eye)
    P = initU(p2, size, eye)
    QP = mx.sym.dot(Q, P)
    #ratio = 100
    #E = ratio * E
    real = mx.sym.dot(QP, mx.sym.dot(eye * mx.sym.cos(E), P, transpose_b=True))
    image = mx.sym.dot(QP, mx.sym.dot(eye * mx.sym.sin(E), P, transpose_b=True))
    return real, image

class Complex:
    def __init__(self, a, b):
        self.a = a#mx.sym.identity(a)
        self.b = b#mx.sym.identity(b)

    def T(self):
        return Complex(mx.sym.transpose(self.a), - mx.sym.transpose(self.b))

    def dot(self, C, left=True):
        if left==True:
            #real = mx.sym.dot(C.a, self.a) -  mx.sym.dot(C.b, self.b)
            #image = mx.sym.dot(C.b, self.a) +  mx.sym.dot(C.a, self.b)
            real = mx.sym.FullyConnected(data=C.a, weight=self.a, num_hidden=256, no_bias=True)
            real -= mx.sym.FullyConnected(data=C.b, weight=self.b, num_hidden=256, no_bias=True)
            image = mx.sym.FullyConnected(data=C.b, weight=self.a, num_hidden=256, no_bias=True)
            image += mx.sym.FullyConnected(data=C.a, weight=self.b, num_hidden=256, no_bias=True)
        else:
            real = mx.sym.dot(self.a, C.a) -  mx.sym.dot(self.b, C.b)
            image = mx.sym.dot(self.a, C.b) +  mx.sym.dot(self.b, C.a)
            print real
        return Complex(real, image)

    def dot_(self, R, left=True):
        if left==True:
            #real = mx.sym.dot(R, self.a)
            #image = mx.sym.dot(R, self.b)
            real = mx.sym.FullyConnected(data=R, weight=self.a, num_hidden=256, no_bias=True)
            image = mx.sym.FullyConnected(data=R, weight=self.b, num_hidden=256, no_bias=True)
        else:
            #real = mx.sym.dot(self.a, R)
            #image = mx.sym.dot(self.b, R)
            real = mx.sym.FullyConnected(data=self.a, weight=R, num_hidden=256, no_bias=True)
            image = mx.sym.FullyConnected(data=self.b, weight=R, num_hidden=256, no_bias=True)

        return Complex(real, image) 
    
    def mult_(self, alpha):
        self.a = alpha * self.a
        self.b = alpha * self.b
        return Complex(self.a, self.b)

    def mult(self, alpha):
        self.a = alpha.a * self.a
        self.b = alpha.b * self.b
        return Complex(self.a, self.b)
    
    def add(self, C):
        a = mx.sym.broadcast_add(self.a, C.a)
        b = mx.sym.broadcast_add(self.b, C.b)
        return Complex(a, b)

    def add_(self, R):
        return Complex(self.a + R, self.b)

    def transform(self, func):
        '''
        def func(a, b):
            a = mx.sym.Activation(a, act_type='relu')
            b = mx.sym.Activation(b, act_type='relu')
            return a, b 
        '''
        a, b = func(self.a, self.b)
        return Complex(a, b)

def ULSTM(num_hidden, indata, prev_state, param, seqidx, layeridx, indata_aux=None, dropout=0.):
    W1 = Complex(param.W1_a, param.W1_b)
    W2 = Complex(param.W2_a, param.W2_b) 
    W3 = Complex(param.W3_a, param.W3_b)
    W4 = Complex(param.W4_a, param.W4_b) 
    W5 = param.W5
    W6 = param.W6
    Bias1 = Complex(param.Bias1_a, param.Bias1_b)
    Bias2 = Complex(param.Bias2_a, param.Bias2_b)

    def func(a, b):
        a = mx.sym.Activation(a, act_type='sigmoid')
        b = mx.sym.Activation(b, act_type='sigmoid')
        return a, b
    
    gate_i2h = W1.dot(indata, left=True).add( W2.dot(prev_state, left=True)).add(Bias1).transform(func)
    gate_h2h = W3.dot(indata, left=True).add( W4.dot(prev_state, left=True)).add(Bias2).transform(func)

    def func_(a, b):  
        a = mx.sym.Activation(a, act_type='tanh')
        b = mx.sym.Activation(b, act_type='tanh')
        return a, b
    gate_residual = indata.dot_(W5, left=False).add( prev_state.dot_(W6, left=False)).transform(func)
    next_h = gate_i2h.mult(indata).add(gate_h2h.mult(prev_state)).transform(func_).add( indata.mult(gate_residual))
    
    return next_h
    '''
    def func__(a,b):
        return mx.sym.L2Normalization(a), mx.sym.L2Normalization(b)
    return next_h.transform(func__)
    '''

def URNN(num_hidden, indata, prev_state, param, seqidx, layeridx, indata_aux=None, dropout=0.):
    W1 = Complex(param.W1_a, param.W1_b)
    W2 = Complex(param.W2_a, param.W2_b)
    Bias = Complex(param.W3_a, param.W3_b)  
    
    if layeridx > -1:
        x1 = W1.dot(indata, left=True) 
    else:
        x1 = W1.dot_(indata, left=True)
    
    x2 = W2.dot(prev_state, left=True)
    
    def func(a, b):
        '''
        a = prev_state.a*0.5 + 0.5*mx.sym.Activation(b, act_type='relu')
        b = prev_state.b*0.5 - 0.5*mx.sym.Activation(a, act_type='relu') 
        '''
        a = mx.sym.Activation(a, act_type='tanh')  
        b = mx.sym.Activation(b, act_type='tanh')  
        return mx.sym.L2Normalization(a), mx.sym.L2Normalization(b)

    if layeridx > 0:
        next_h = x1.add(x2).add(Bias).transform(func)#.mult_(1.1)#.add(indata)
    else:
        next_h = x1.add(x2).add(Bias).transform(func)#.mult_(1.1)#.add_(indata)
    return next_h

def URNN_unroll(seq_len, input_size, num_layers,                              
                     num_hidden, num_embed, dropout=0.): 
    data = [mx.sym.Variable('data'), mx.sym.Variable('data_aux')]
    embed_weight = [mx.sym.Variable("embed_weight"), mx.sym.Variable("aux_embed_weight")]
    embed = [mx.sym.Embedding(data=data[0], input_dim=input_size[0],
                             weight=embed_weight[0], output_dim=num_embed[0], name='embed'),
            mx.sym.Embedding(data=data[1], input_dim=input_size[1],
                            weight=embed_weight[1], output_dim=num_embed[1], name='aux_embed')]
    
    l1_U0a, l1_U0b = initUMatrix("l1_U0", num_hidden)
    l1_U1a, l1_U1b = initUMatrix("l1_U1", num_hidden)
    l1_U2a, l1_U2b = initUMatrix("l1_U2", num_hidden)
    l1_U3a, l1_U3b = initUMatrix("l1_U3", num_hidden)
    l1_U4a, l1_U4b = initUMatrix("l1_U4", num_hidden)
    l1_U5a, l1_U5b = initUMatrix("l1_U5", num_hidden)
    
    l0_U0a, l0_U0b = initUMatrix("l0_U0", num_hidden)
    l0_U1a, l0_U1b = initUMatrix("l0_U1", num_hidden)
    l0_U2a, l0_U2b = initUMatrix("l0_U2", num_hidden)
    l0_U3a, l0_U3b = initUMatrix("l0_U3", num_hidden)
    l0_U4a, l0_U4b = initUMatrix("l0_U4", num_hidden)
    l0_U5a, l0_U5b = initUMatrix("l0_U5", num_hidden)

    embed_aux_weight = mx.sym.Variable(name='aux_in_weight') 
    embed_ = mx.sym.Embedding(data=data[0], input_dim=input_size[0],
                            weight=embed_aux_weight, output_dim=num_embed[0]+num_embed[1], name='embed_')

    urnn_param = [ULSTMParam(W1_a=l1_U0a,
                              W1_b=l1_U0b,
                              W2_a=l1_U1a,
                              W2_b=l1_U1b,
                              W3_a=l1_U2a,
                              W3_b=l1_U2b,
                              W4_a=l1_U3a,
                              W4_b=l1_U3b,
                              W5=mx.sym.Variable("l0_W5_weight",shape=(num_hidden,num_hidden)),
                              W6=mx.sym.Variable("l0_W6_weight",shape=(num_hidden,num_hidden)),
                              Bias1_a=mx.sym.Variable("l0_bias1a_weight",shape=(1,num_hidden)),
                              Bias1_b=mx.sym.Variable("l0_bias1b_weight",shape=(1,num_hidden)),
                              Bias2_a=mx.sym.Variable("l0_bias2a_weight",shape=(1,num_hidden)),
                              Bias2_b=mx.sym.Variable("l0_bias2b_weight",shape=(1,num_hidden))
                              ),

                ULSTMParam(W1_a=l0_U0a,
                              W1_b=l0_U0b,
                              W2_a=l0_U1a,
                              W2_b=l0_U1b,
                              W3_a=l0_U2a,
                              W3_b=l0_U2b,
                              W4_a=l0_U3a,
                              W4_b=l0_U3b,
                              W5=mx.sym.Variable("l1_W5_weight",shape=(num_hidden,num_hidden)),
                              W6=mx.sym.Variable("l1_W6_weight",shape=(num_hidden,num_hidden)),
                              Bias1_a=mx.sym.Variable("l1_bias1a_weight",shape=(1,num_hidden)),
                              Bias1_b=mx.sym.Variable("l1_bias1b_weight",shape=(1,num_hidden)),
                              Bias2_a=mx.sym.Variable("l1_bias2a_weight",shape=(1,num_hidden)),
                              Bias2_b=mx.sym.Variable("l1_bias2b_weight",shape=(1,num_hidden))
                              )
                              ]

    c = mx.sym.Variable("LSTM_state_cell")
    h = mx.sym.Variable("LSTM_state")
    last_states = Complex(h, c)
    #------------------------------------------------------------------------
    embed = mx.sym.Concat(*[embed[0],embed[1]], dim=2)
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    wordvec_ = mx.sym.SliceChannel(data=embed_, num_outputs=seq_len, squeeze_axis=1)
    wordvec = [Complex(wordvec[i], wordvec_[i]) for i in range(seq_len)]

    seqidxs_ = [i for i in range(seq_len)]
    _seqidxs = [seq_len - i - 1 for i in range(seq_len)]
    forward = True
    hidden = ['' for i in range(seq_len)]
    hidden_a = ['' for i in range(seq_len)] 
    input_layer = wordvec  #layer0


    for layeridx in range(num_layers):
        if forward == True:
            forward_param = urnn_param[0]
            seqidxs = seqidxs_
        else:
            forward_param = urnn_param[1]
            seqidxs = _seqidxs
        for seqidx in seqidxs:
            indata_aux = None  
            next_state = ULSTM(num_hidden, indata=input_layer[seqidx],
                              prev_state=last_states,
                              param=forward_param,
                              seqidx=seqidx, layeridx=layeridx,
                              indata_aux = indata_aux, dropout=dropout)
            last_states = next_state
            hidden[seqidx] = next_state
            '''
            if layeridx > -1:
                hidden[seqidx] = next_state.add(input_layer[seqidx])
            else:
                hidden[seqidx] = next_state
            '''
            hidden_a[seqidx] = mx.sym.Concat(*[next_state.a, next_state.b], dim=1)
        input_layer = hidden
        forward = not forward
    #-----------------------------------------------------
    src_codes = mx.sym.Concat(*hidden_a, dim=1)    
    if dropout > 0.:
        src_codes = mx.sym.Dropout(data=src_codes, p=dropout)
    return src_codes


def EG_lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, indata_aux=None, dropout=0.):
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 3,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    c2h = mx.sym.FullyConnected(data=prev_state.c,
                                weight=param.h2h_weight,
                                bias=param.aux2h_bias,
                                num_hidden=num_hidden * 3,
                                name="t%d_l%d_aux2h" % (seqidx, layeridx))  
    '''
    h2h = mx.sym.FullyConnected(data=prev_state.h,  
                                weight=param.aux2h_weight,
                                bias=param.aux2h_bias,   
                                num_hidden=num_hidden * 3,
                                name="t%d_l%d_aux2h" % (seqidx, layeridx))
    '''

    gates = i2h + c2h

    slice_gates = mx.sym.SliceChannel(gates, num_outputs=3,                      
                                                    name="t%d_l%d_slice" % (seqidx, layeridx))

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    forget_gate = mx.sym.Activation(slice_gates[1], act_type="sigmoid")
    mix_gate  =  mx.sym.Activation(slice_gates[2], act_type="sigmoid")

    next_c = in_gate * indata + forget_gate * prev_state.c 
    next_h = mx.sym.Activation(next_c, act_type="tanh")  + mix_gate * indata + 0* prev_state.h
    return LSTMState(c=next_c, h=next_h)
    '''
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 5,
                            	name="t%d_l%d_h2h" % (seqidx, layeridx))
    
    if indata_aux is not None:
        aux2h = mx.sym.FullyConnected(data=indata_aux,
                                    weight=param.aux2h_weight,
                                    bias=param.aux2h_bias,
                                    num_hidden=num_hidden * 5,
                                    name="t%d_l%d_aux2h" % (seqidx, layeridx))
    gates = i2h + h2h
    
    if indata_aux is not None:
        gates += aux2h

    slice_gates = mx.sym.SliceChannel(gates, num_outputs=5,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))

    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    
    mix_gate  =  mx.sym.Activation(slice_gates[4], act_type="sigmoid")

    next_c = forget_gate * prev_state.c + in_gate * in_transform
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh") + mix_gate * indata 

    return LSTMState(c=next_c, h=next_h)
    '''
def bi_lstm_unroll(seq_len, input_size, num_layers,
                num_hidden, num_embed, dropout=0.):

    data = [mx.sym.Variable('data'), mx.sym.Variable('data_aux')]
    embed_weight = [mx.sym.Variable("embed_weight"), mx.sym.Variable("aux_embed_weight")]
    embed = [mx.sym.Embedding(data=data[0], input_dim=input_size[0],
                             weight=embed_weight[0], output_dim=num_embed[0], name='embed'),
            mx.sym.Embedding(data=data[1], input_dim=input_size[1],
                            weight=embed_weight[1], output_dim=num_embed[1], name='aux_embed')]

    embed_aux_weight = mx.sym.Variable(name='aux_in_weight')
    lstm_param = [LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"),
                              aux2h_weight=mx.sym.Variable("l0_aux2h_weight"),
                              aux2h_bias=mx.sym.Variable("l0_aux2h_bias")),

                LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"),
                              aux2h_weight=mx.sym.Variable("l1_aux2h_weight"),
                              aux2h_bias=mx.sym.Variable("l1_aux2h_bias")
                              )]

    last_states = LSTMState(c = mx.sym.Variable("LSTM_state_cell"),
                            h = mx.sym.Variable("LSTM_state"))
    init_state =  last_states
    #------------------------------------------------------------------------
    embed = mx.sym.Concat(*[embed[0],embed[1]], dim=2)
    wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    seqidxs_ = [i for i in range(seq_len)]
    _seqidxs = [seq_len - i - 1 for i in range(seq_len)]
    forward = True
    hidden = [[None for i in range(seq_len)]]*2 # ['' for i in range(seq_len)]
    input_layer = wordvec  #layer0

    for layeridx in range(num_layers):
        if forward == True:
            forward_param = lstm_param[0]
            seqidxs = seqidxs_
        else:
            forward_param = lstm_param[1]
            seqidxs = _seqidxs
        hidden.append([None for i in range(seq_len)])
        #last_states = init_state
        for seqidx in seqidxs:
            '''
            if forward:
                if seqidx > 0:
                #if seqidx < seq_len - 1:
                    indata_aux = hidden[layeridx][seqidx - 1]
                else:
                    indata_aux = None
            else:
                if seqidx < seq_len - 1:
                    indata_aux = hidden[layeridx][seqidx + 1] 
                else:
                    indata_aux = None
            '''
            '''
            if forward == False:
                memory = mx.sym.Reshape(mx.sym.Concat(*input_layer, dim=1), shape=(-1, seq_len, num_hidden))
                key = mx.sym.expand_dims(last_states.h, axis=2)
                attention = mx.sym.softmax(mx.sym.batch_dot(memory, key), axis=1)
                atten_vec = mx.sym.batch_dot(memory, attention, transpose_a =True)
                indata_aux = mx.sym.flatten(atten_vec)
            else:
                indata_aux = None 
            '''
            indata_aux = None  
            next_state = EG_lstm(num_hidden, indata=input_layer[seqidx],
                              prev_state=last_states,
                              param=forward_param,
                              seqidx=seqidx, layeridx=layeridx,
                              indata_aux = indata_aux, dropout=dropout)
            last_states = next_state
            #print hidden, layeridx
            hidden[layeridx+2][seqidx] = next_state.h
        input_layer = hidden[layeridx+2]
        forward = not forward
    #-----------------------------------------------------
    return mx.sym.Concat(*(hidden[num_layers - 1 +2]), dim=1)

def predict_label_wapper(rnn,seq_len,num_hidden,num_label,idx=''):
    cls_weight = mx.sym.Variable("id_%s_cls_weight"%idx)
    cls_bias = mx.sym.Variable("id_%s_cls_bias"%idx)
    pred = mx.sym.FullyConnected(data=rnn, num_hidden=num_label,
                                weight=cls_weight, bias=cls_bias, name='id_%s_pred'%idx)
    pred_tm = mx.sym.Reshape(data=pred, shape=(-1 , seq_len,  num_label))
    result = mx.sym.softmax(data=pred_tm,axis=2)
    loss0_ng = mx.sym.softmax(data=-1.0*pred_tm,axis=2)

    loss1 = mx.sym.smooth_l1(pred_tm ,scalar=0.5) * 0.05
    return result + loss1 - loss0_ng

def tmRnn_loss_wapper(rnn,seq_len,num_hidden,num_label,label,idx=''):
    cls_weight = mx.sym.Variable("id_%s_cls_weight"%idx)
    cls_bias = mx.sym.Variable("id_%s_cls_bias"%idx)
    #gate_weight = mx.sym.Variable("id_%s_l0_gate_weight"%idx) 
    pred = mx.sym.FullyConnected(data=rnn, num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='id_%s_pred'%idx)
    #gate = mx.sym.FullyConnected(data=rnn, num_hidden=1,
    #                             weight=gate_weight,no_bias=True, name='id_%s_gate'%idx)
    # reshape to be of compatible shape as labels
    pred_tm = mx.sym.Reshape(data=pred, shape=(-1 , seq_len,  num_label))
    #gate = mx.sym.Reshape(data=gate, shape=(-1 , seq_len, 1))

    loss0 = mx.sym.SoftmaxOutput(data=pred_tm,label=label, preserve_shape=True,
                              name='id_%s_softmax'%idx)
    '''
    pred1 = -1.0*pred_tm
    pred_tm1 = mx.sym.Reshape(data=pred1, shape=(-1 , seq_len,  num_label)) 
    loss0_ng = mx.sym.SoftmaxOutput(data=pred_tm,label=label, preserve_shape=True,
                                name='id_%s_softmax_ng'%idx)
    '''
    #loss1 = mx.sym.smooth_l1(pred_tm ,scalar=0.5) * 0.05
    loss1 =  mx.sym.smooth_l1(pred_tm ,scalar=0.5, normalization='batch') * 0.05 
    loss1 = mx.sym.MakeLoss(loss1)

    return mx.sym.Group([mx.sym.BlockGrad(loss0), loss0 + loss1])
