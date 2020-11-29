from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import json
import argparse

import sys
sys.path.insert(0, 'src')
from nets import SimpleTransformNet

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
parser.add_argument('--modeltype', dest='modeltype', default='fnst', help='type of model')
args = parser.parse_args()


class Exporter():

	def __init__(self, dataset_name = "terrain", type = "fnst"):
		self.dataset_name = dataset_name
		self.type = type
		self.load_model()
		self.export()

	def load_model(self):
		net = SimpleTransformNet()
		self.loaded_model = net.model
		# load weights into new model
		self.loaded_model.load_weights("model/" + self.dataset_name + "_weight.h5")
		print(self.loaded_model.to_json())

	def export(self):
		weights_dict = []
		weights_list = self.loaded_model.get_weights()
		for i, weights in enumerate(weights_list):
			thisweight = {}
			thisweight["shape"] = weights.shape
			if len(weights.shape) == 1:
				thisweight["arrayweight"] = weights.tolist()
			else:
				thisweight["kernelweight"] = weights.tolist()
			weights_dict.append(thisweight)
		formatstr = '{"model":%s,"weights":%s}' % (self.loaded_model.to_json(), json.dumps(weights_dict))
		with open("../NNPostProcessing/Assets/Script/RawModel/%s.json" % self.dataset_name, "w") as json_file:
		    json_file.write(formatstr)
		print(self.loaded_model.summary())
		#plot_model(self.loaded_model, to_file='model/' + self.dataset_name + '_architecture.png', show_shapes=True)

if __name__ == '__main__':
	Exporter(dataset_name = args.dataset_name, type = args.modeltype)