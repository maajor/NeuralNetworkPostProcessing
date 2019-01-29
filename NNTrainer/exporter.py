from keras.models import model_from_json, Model
from keras.utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from data_loader import DataLoader
import json
import argparse

import sys
sys.path.insert(0, 'src')
import nets

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
parser.add_argument('--modeltype', dest='modeltype', default='pix2pix', help='type of model')
args = parser.parse_args()


class Exporter():

	def __init__(self, dataset_name = "terrain", type = "pix2pix"):
		self.dataset_name = dataset_name
		self.type = type
		self.load_model()
		self.export()

	def load_model(self):
		if self.type == "pix2pix":
			json_file = open("model/" + self.dataset_name + "_architecture.json", 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			self.loaded_model = model_from_json(loaded_model_json)
		if self.type == "fnst":
			net = nets.image_transform_net_simple(512,512,3)
			net.compile(Adam(),  loss='mse')
			self.loaded_model = net
		# load weights into new model
		self.loaded_model.load_weights("model/" + self.dataset_name + "_weight.h5")

	def export(self):
		weights_dict = []
		weights_list = self.loaded_model.get_weights()
		for i, weights in enumerate(weights_list):
			thisweight = {}
			thisweight["shape"] = weights.shape;
			if len(weights.shape) == 1:
				thisweight["arrayweight"] = weights.tolist()
			else:
				thisweight["kernelweight"] = weights.tolist()
			weights_dict.append(thisweight)
		formatstr = '{"model":%s,"weights":%s}' % (self.loaded_model.to_json(), json.dumps(weights_dict))
		with open("../NNPostProcessing/Assets/Script/RawModel/%s.json" % self.dataset_name, "w") as json_file:
		    json_file.write(formatstr)
		print(self.loaded_model.summary())
		plot_model(self.loaded_model, to_file='model/' + self.dataset_name + '_architecture.png', show_shapes=True)

if __name__ == '__main__':
	Exporter(dataset_name = args.dataset_name, type = args.modeltype)