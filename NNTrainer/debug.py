from keras.models import model_from_json, Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.utils import plot_model
import matplotlib.pyplot as plt
from data_loader import DataLoader
import json
import time
import argparse
from scipy.misc import imsave

import sys
sys.path.insert(0, 'src')
import nets

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
parser.add_argument('--modelpath', dest='modelpath', default='model', help='name of model')
parser.add_argument('--modeltype', dest='type', default='pix2pix', help='model type')
args = parser.parse_args()

class DebuggerFST():
	def __init__(self, path = "model", dataset_name = "terrain"):
		self.path = path
		self.dataset_name = dataset_name
		self.load_model()
		self.plot()
		self.debug()

	def load_model(self):
		net = nets.image_transform_net_simple(512,512,3)
		net.compile(Adam(),  loss='mse')
		net.load_weights("model/"+self.path+'_weight.h5',by_name=False)
		#net.summary()
		#predictnet = Model(input = model.input, output = model.get_layer(name = "transform_output").output)
		self.loaded_model = net
		'''
		json_file = open("model/" + self.path + "_architecture.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("model/" + self.path + "_weight.h5")'''

	def plot(self):
		plot_model(self.loaded_model, to_file="model/" + self.path +"_architecture.png", show_shapes=True)
		print("Saved model to disk")

	def debug(self):
		fig, axs = plt.subplots(1,1, figsize=(15,15))
		data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(512, 512))
		imgs_A, imgs_B = data_loader.load_data(batch_size=1, is_testing= False, is_debug = True)
		inputimg = (imgs_B[:,:,:,:3] + 1 ) * 0.5
		imsave("images/" + self.dataset_name + "/source.png", inputimg[0])
		imgs_B = (imgs_B[:,:,:,:3] + 1 ) * 127.5
		fake_A = self.loaded_model.predict(imgs_B[:,:,:,:3])
		t1 = time.time()
		fake_A = self.loaded_model.predict(imgs_B[:,:,:,:3])
		print("process: %s" % (time.time() -t1))
		fake_A =fake_A / 255
		imsave("images/" + self.dataset_name + "/predict.png", fake_A[0])
		'''
		for layernum in range(1, len(self.loaded_model.layers)):
			intermediate_model = Model(inputs = self.loaded_model.input, outputs = self.loaded_model.layers[layernum].output)
			inter_output = intermediate_model.predict(imgs_B)
			inter_output = inter_output * 0.5 + 0.5
			#imsave("images/" + self.dataset_name + "/inter" + str(layernum) + ".png", inter_output[0,:,:,0:3])
			axs.imshow(inter_output[0,:,:,0:3])
			fig.savefig("images/" + self.dataset_name + "/inter" + str(layernum) + ".png")'''

class Debugger():

	def __init__(self, path = "model", dataset_name = "terrain"):
		self.path = path
		self.dataset_name = dataset_name
		self.load_model()
		self.debug()

	def load_model(self):
		json_file = open("model/" + self.path + "_architecture.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("model/" + self.path + "_weight.h5")

	def debug(self):
		self.loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


		data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(512, 512))
		imgs_A, imgs_B = data_loader.load_data(batch_size=1,  is_testing= False, is_debug = True)
		interresult0 = imgs_B.tolist()
		t1 = time.time()
		fake_A = self.loaded_model.predict(imgs_B)
		print("process: %s" % (time.time() -t1))
		fake_A = 0.5 * fake_A + 0.5
		fig, axs = plt.subplots(1,1, figsize=(15,15))
		axs.imshow(fake_A[0])
		fig.savefig("images/" + self.dataset_name + "/predict.png")
		axs.imshow(0.5 * imgs_B[0] + 0.5)
		fig.savefig("images/" + self.dataset_name + "/source.png")

		with open("debug/intermediate_result0.json", "w") as json_file:
		    	json_file.write(json.dumps(interresult0))

		for layernum in range(1, len(self.loaded_model.layers)):
			intermediate_model = Model(inputs = self.loaded_model.input, outputs = self.loaded_model.layers[layernum].output)
			inter_output = intermediate_model.predict(imgs_B)
			with open("debug/intermediate_result" + str(layernum) + ".json", "w") as json_file:
				json_file.write(json.dumps(inter_output.tolist()))
			inter_output = 0.5 * inter_output + 0.5
			axs.imshow(inter_output[0,:,:,0:3])
			fig.savefig("images/" + self.dataset_name + "/inter" + str(layernum) + ".png")


if __name__ == '__main__':
	if args.type == "pix2pix":
		Debugger(path = args.modelpath, dataset_name = args.dataset_name)
	if args.type == "fst":
		DebuggerFST(path = args.modelpath, dataset_name = args.dataset_name)