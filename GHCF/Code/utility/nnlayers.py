import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.01

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	# else:
	# 	print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)
#变量初始化器为xavier 使用get——variable创建变量
def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	#断言 判断对错 错误结束 并输出
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean)
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret
#FC 输出为[inp第二维度，outdim]
def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer='zeros', reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu':
		temMask = tf.to_float(tf.greater(data, 1.0))
		ret = temMask * (1 + leaky * (data - 1)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

def multiHeadAttention(localReps, glbRep, number, numHeads, inpDim):
	query = tf.reshape(tf.tile(tf.reshape(FC(glbRep, inpDim, useBias=True, reg=True), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
	temLocals = tf.reshape(localReps, [-1, inpDim])
	key = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, numHeads, inpDim//numHeads])
	val = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, number, numHeads, inpDim//numHeads])
	att = tf.nn.softmax(tf.reshape(tf.reduce_sum(query * key, axis=-1), [-1, number, numHeads, 1]), axis=1)
	attRep = tf.reshape(tf.reduce_sum(val * att, axis=1), [-1, inpDim])
	return attRep

# def selfAttention(localReps, number, inpDim, numHeads):
# 	attReps = [None] * number
# 	stkReps = tf.stack(localReps, axis=1)
# 	for i in range(number):
# 		glbRep = localReps[i]
# 		temAttRep = multiHeadAttention(stkReps, glbRep, number=number, numHeads=numHeads, inpDim=inpDim) + glbRep
# 		fc1 = FC(temAttRep, inpDim, reg=True, useBias=True, activation='relu') + temAttRep
# 		fc2 = FC(fc1, inpDim, reg=True, useBias=True, activation='relu') + fc1
# 		fc2 = temAttRep
# 		attReps[i] = fc2
# 	return attReps
# localReps catlat1, number 行为个数4, inpDim 嵌入维度16, numHeads 注意力头数目2
def selfAttention(localReps, number=3, inpDim=64, numHeads=2):
	#Q，K，V [16,16]
	Q = defineRandomNameParam([inpDim, inpDim], reg=True)
	K = defineRandomNameParam([inpDim, inpDim], reg=True)
	V = defineRandomNameParam([inpDim, inpDim], reg=True)
	#rspreps [?,16]
	rspReps = tf.reshape(tf.stack(localReps, axis=1), [-1, inpDim])
	#q: [-1,3,1,2,32]
	#k: [-1,1,3,2,32]
	#v: [-1,1,3,2,32]
	q = tf.reshape(rspReps @ Q, [-1, number, 1, numHeads, inpDim//numHeads])
	k = tf.reshape(rspReps @ K, [-1, 1, number, numHeads, inpDim//numHeads])
	v = tf.reshape(rspReps @ V, [-1, 1, number, numHeads, inpDim//numHeads])
	# att [-1,3,3,2,32]->[-1,3,1,2,32]
	att = tf.nn.softmax(tf.reduce_sum(q * k, axis=-1, keepdims=True) / (inpDim/numHeads), axis=2)
	# attval: [-1,3,1,2,32]-->[-1,3,64]
	attval = tf.reshape(tf.reduce_sum(att * v, axis=2), [-1, number, inpDim])
	rets = [None] * number
	#slice 
	for i in range(number):
		rets[i] = tf.reshape(tf.slice(attval, [0, i, 0], [-1, 1, -1]), [-1, inpDim])
	return rets