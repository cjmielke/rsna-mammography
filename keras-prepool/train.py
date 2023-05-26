

import pandas as pd
import logging
import argparse
import os
import numpy as np
#from tensorflow.keras.legacy import interfaces
import tensorflow
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras.callbacks import Callback

from accadam import AccAdam
from data import labels
from models import buildModel
from sklearn.model_selection import train_test_split

from generators import ImageGen
import kerop

def get_args(bs=8, act='relu', fc1=128, fc2=64):
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('-encoder', default='vgg19', type=str)
	parser.add_argument('-flat', default='max', type=str)
	parser.add_argument('-opt', default='adam', type=str)
	parser.add_argument('-activation', default=act, type=str)
	parser.add_argument('-size', default=1024, type=int)
	parser.add_argument('-steps', default=500, type=int)
	parser.add_argument('-fc0', default=0, type=int)
	parser.add_argument('-fc1', default=fc1, type=int)
	parser.add_argument('-fc2', default=fc2, type=int)
	parser.add_argument('-bs', default=bs, type=int)
	parser.add_argument('-bn', default=1, type=int)
	parser.add_argument('-dropout', default=0.0, type=float)
	parser.add_argument('-weights', default=None, type=str)
	parser.add_argument('-classifiers', default=None, type=str, help='Load in pre-trained final output layers')
	parser.add_argument('-prepool', action='store_true')
	parser.add_argument('-freezeEncoder', action='store_true')
	parser.add_argument('-convSize', default=3, type=int)
	parser.add_argument('-filters', default=32, type=int)
	parser.add_argument('-doubleLayers', action='store_true')
	parser.add_argument('-trainSeparately', action='store_true')
	parser.add_argument('-aux1', action='store_true')
	args = parser.parse_args()
	return args



class autoEncoderImages(Callback):
	def on_epoch_end(self, epoch, logs=None):
		pass





def getModel(args):
	model, preproc = buildModel(args)

	#from shampoo_optimizer import ShampooOptimizer
	#opt = ShampooOptimizer(learning_rate=0.001)
	opt = SGD(lr=0.001)
	#opt = SGD(lr=0.0001)
	#opt = SGD(lr=0.000001)
	#opt = SGD(lr=0.00000001)

	# opt = AccAdam(accumulation_steps=16)
	# from keras_gradient_accumulation import GradientAccumulation
	# opt = GradientAccumulation('adam', accumulation_steps=8)
	losses = {k: 'binary_crossentropy' for k in labels}
	if args.aux1:
		for lbl in labels: losses['aux1_'+lbl] = 'binary_crossentropy'
	#losses['oi'] = 'mse'
	lossweights = {k: 1 for k in labels}

	# lossweights['Pneumothorax'] = 1.0
	# for l in ['oi2']: losses[l] = 'mse'
	metrics = {}
	#metrics.append(tensorflow.keras.metrics.AUC(from_logits=True))
	if len(labels) == 1:
		metrics[labels[0]] = 'accuracy'

	model.compile(
		#optimizer=args.opt,
		optimizer=opt,
		# optimizer=sgd,
		# loss=dict(age="categorical_crossentropy", gender="categorical_crossentropy", decade="categorical_crossentropy", old='categorical_crossentropy'),
		# loss_weights=dict(age=0.0, gender=0.0, decade=0.0, old=1.0),
		# loss=dict(age="binary_crossentropy", gender="categorical_crossentropy", old='categorical_crossentropy'),
		loss=losses,
		#loss_weights=lossweights,
		# loss = 'binary_crossentropy',
		# metrics={k: 'accuracy' for k in labels},
		metrics=metrics
	)

	if args.weights:
		model.load_weights(args.weights, by_name=True)

	#layer_name, layer_flops, inshape, weights = kerop.profile(model)

	# visualize results
	#for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
	#	print("layer:", name, shape, " MegaFLOPS:", flop / 1e6, " Weights:", weight )
	#for name, flop, shape in zip(layer_name, layer_flops, inshape):
	#	print("layer:", name, shape, " MegaFLOPS:", flop / 1e6)

	return model, preproc

def main(args):
	# Open a strategy scope.
	strategy = tensorflow.distribute.MirroredStrategy()
	#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	with strategy.scope():
		model, preproc = getModel(args)

	#stopEarly = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=30, verbose=1)

	callbacks = [
		#stopEarly,
		#LearningRateScheduler(schedule=Schedule(nb_epochs)),
	]

	if False:
		callbacks.append(ModelCheckpoint( "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5" ,
			monitor="val_loss",
			verbose=1,
			save_best_only=True,
			mode="auto"))

	if False:
		from tensorflow.keras.callbacks import ReduceLROnPlateau
		#pl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
		# ^ defaults
		pl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='auto', cooldown=0, min_lr=0)
		callbacks.append(pl)


	from generators import ImageGen
	df = pd.read_csv('data.csv').fillna(0)#.sample(frac=1)
	#df = df.head(100)   # FIXME
	trainDF, valDF = train_test_split(df, train_size=0.9)
	print(trainDF.shape, valDF.shape)
	IG = ImageGen(trainDF, preproc, batchSize=args.bs, size=args.size, aug=False, steps=args.steps, aux1=args.aux1)
	IGVal = ImageGen(valDF, preproc, batchSize=args.bs, size=args.size, aug=False, steps=args.steps//10, aux1=args.aux1)
	if args.encoder == 'custom' or args.prepool:
		IG.ncolors = 1
		IGVal.ncolors = 1

	try:
		model.fit(
			#stratifiedQ.trainGen,
			IG,
			shuffle=False,
			verbose=1,
			epochs=9999,
			#samples_per_epoch=batch_size,
			#steps_per_epoch=train_num / batch_size,
			#steps_per_epoch=100,
			callbacks=callbacks,
			#validation_data=valBatcher.batchGen,
			validation_data=IGVal,
			#validation_steps=10,
			max_queue_size=100,
			use_multiprocessing=False#, workers=4
		)
	except KeyboardInterrupt:
		pass

	while True:
		try:
			model.save('model.h5')
			break
		except KeyboardInterrupt:
			print('saving model - wait')

def trainSep(args):

	model, subModels, preproc = buildModel(args)

	opt = SGD(lr=0.001)
	for model in subModels:
		model.compile(
			optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']
		)

	if args.weights:
		model.load_weights(args.weights, by_name=True)

	df = pd.read_csv('data.csv').fillna(0)#.sample(frac=1)
	#df = df.head(100)   # FIXME
	trainDF, valDF = train_test_split(df, train_size=0.9, random_state=42)

	generators = []
	for idx, label in enumerate(labels):
		IG = ImageGen(trainDF, preproc, batchSize=args.bs, size=args.size, aug=False, steps=args.steps//(idx+1), stratLabels=[label])
		IGVal = ImageGen(valDF, preproc, batchSize=args.bs, size=args.size, aug=False, steps=10, stratLabels=[label])
		if args.encoder=='custom' or args.prepool:
			IG.ncolors = 1
			IGVal.ncolors = 1
		generators.append((IG, IGVal))

	try:
		while True:
			for model, gens, label in zip(subModels, generators, labels):
				print('                         '+label+'            ')
				IG, IGVal = gens
				model.fit(
					#stratifiedQ.trainGen,
					IG,
					shuffle=False, verbose=1, epochs=1,
					#samples_per_epoch=batch_size,
					#steps_per_epoch=train_num / batch_size,
					#steps_per_epoch=100,
					#validation_data=valBatcher.batchGen,
					validation_data=IGVal,
					#validation_steps=10,
					max_queue_size=100
				)
	except KeyboardInterrupt:
		pass

	#model.save(OUTDIR + 'model.h5')
	model.save('model.h5')


if __name__ == '__main__':
	args = get_args()
	if args.trainSeparately:
		trainSep(args)
	else:
		main(args)

