import keras
import numpy as np 
import matplotlib.pyplot as plt
from metrics import acc
#from keras.models import load_model # <- CAN'T! Have to reinit model and load weights
from datasets import load_mnist, load_usps, load_fashion_mnist
from DCEC import DCEC
from ConvAE import CAE
from keras.models import Sequential, Model
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import classification_report

if __name__ == "__main__":
	# setting the hyper parameters
	import argparse
	parser = argparse.ArgumentParser(description='train')
	parser.add_argument('dataset', default='mnist', choices=['mnist', 'usps', 'mnist-test', 'fashion', 'cifar10'])
	args = parser.parse_args()
	print(args)

	dataset = args.dataset  # ['mnist', 'fashion']

	if dataset == 'mnist':
		x, y = load_mnist()
	elif dataset == 'fashion':
		x, y = load_fashion_mnist()
	elif dataset == 'cifar10': # <- 10 classes
		x, y = load_cifar10()
	print(x.shape)
	print(y.shape)
	print(np.unique(y))

	# check if balanced
	for i in np.unique(y):
		print(i, sum(y == i))
	# -> yes, somewhat

	#model_path = "/home/andrep/workspace/clustering/DCEC/results/temp_050120/dcec_model_0.h5"
	model_path = "/home/andrep/workspace/clustering/DCEC/results/temp_" + dataset + "_050120/dcec_model_final.h5"
	#model_path = "/home/andrep/workspace/clustering/DCEC/results/temp_050120/pretrain_cae_model.h5"

	# initialize model
	dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, 10], n_clusters=10)
	dcec.load_weights(model_path)
	model = dcec.model


	# pred with all values
	res = dcec.predict(x)

	# predict that most represented number in cluster represent it's prediction
	preds = np.zeros_like(res)
	for i in range(10):
		tmp = y[res == i]
		print(i)
		print(tmp)
		print(np.median(tmp))
		preds[res == i] = np.median(tmp) # <- burde vært typetall

	# -> actual unsupervised classification accuracy (after supervision of clusters)
	print(classification_report(y, preds))

	# -> clustering performance metrics (unsupervised)
	print(acc(y, res))
	print(normalized_mutual_info_score(y, res))
	print(adjusted_rand_score(y, res))
	# -> should, in some way, reflect classification performance as well(?)

	x_val = 6
	y_val = 10

	while True:

		fig, ax = plt.subplots(x_val, y_val, figsize=(26, 16))
		plt.tight_layout()
		imgs = []

		# plot random samples from each cluster
		for i in range(y_val):
			# x_tmp = x[res == i] 
			x_tmp = x[y == i]
			np.random.shuffle(x_tmp)
			for j in range(x_val):
				imgs.append(x_tmp[j])
				ax[j, i].imshow(x_tmp[j, ..., 0], cmap="gray")
				ax[j, i].axis('off')

		fig2, ax2 = plt.subplots(x_val, y_val, figsize=(26, 16))
		plt.tight_layout()

		cnt = 0
		for i in range(y_val):
			for j in range(x_val):
				res1, res2 = model.predict(np.expand_dims(imgs[cnt], axis=0), verbose=0)
				ax2[j, i].imshow(res2[0, ..., 0])
				ax2[j, i].axis('off')

				cnt += 1

		plt.show()



	exit()

	# extract features
	#feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
	'''
	while True:
		cnt = np.random.randint(x.shape[0])
		#features = feature_model.predict(np.expand_dims(x[cnt], axis=0))
		#print(features)
		print(dcec.predict(np.expand_dims(x[cnt], axis=0)))
		print(y[cnt])
		print()
		plt.pause(1)
	print(dcec.y_pred)
	exit()
	'''

	# freeze all layers in model during inference -> equivalent to compile=False ?
	for layer in model.layers:
		layer.trainable = False
	# -> sadly, didn't really change anything...

	#model.set_weights(model_path)

	#cae = CAE(x.shape[1:], [32, 64, 128, 10])
	#hidden = cae.get_layer(name='embedding').output
	#encoder = Model(inputs=cae.input, outputs=hidden)

	# do the same with the clustering layer
	#kmeans = KMeans(n_clusters=10, n_init=20)
	#y_pred = kmeans.fit_predict(encoder.predict(x))

	#exit()
	#y_pred_last = np.copy(self.y_pred)
	#self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

	print(model.summary())

	print(model.get_layer(name='clustering').get_config())
	print(model.get_layer(name='clustering').get_weights())

	#exit()

	fig, ax = plt.subplots(1, 2)
	while True:
		cnt = np.random.randint(x.shape[0])
		res1, res2 = model.predict(np.expand_dims(x[cnt], axis=0), verbose=0)
		ax[0].imshow(x[cnt, ..., 0])
		ax[1].imshow(res2[0, ..., 0])
		ax[0].set_title("Pred: " + str(dcec.predict(np.expand_dims(x[cnt], axis=0))[0]) + ", GT: " + str(y[cnt]))
		ax[1].set_title("Reconstructed")
		plt.pause(2)
	plt.show()

	exit()

	x_val = 6
	y_val = 4

	fig, ax = plt.subplots(y_val, x_val, figsize=(20, 13))
	while True:
		for i in range(x_val):
			for j in range(y_val):
				cnt = np.random.randint(x.shape[0])
				ax[j, i].imshow(x[cnt, ..., 0], cmap="gray")
				print(y[cnt])
				#print(model.predict(np.expand_dims(x[cnt], axis=0), verbose=0))
				print(dcec.predict(np.expand_dims(x[cnt], axis=0)))
				ax[j, i].set_title("Pred: " + str(dcec.predict(np.expand_dims(x[cnt], axis=0))[0]) + ", GT: " + str(y[cnt]))
		plt.pause(2)
	plt.show()