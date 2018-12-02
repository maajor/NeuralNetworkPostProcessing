from keras.models import model_from_json, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from data_loader import DataLoader
import json

class Exporter():

	def __init__(self, path = "model", dataset_name = "terrain"):
		self.path = path
		self.dataset_name = dataset_name
		self.load_model()
		self.export()

	def load_model(self):
		json_file = open("model/" + self.path + "_architecture.json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		self.loaded_model.load_weights("model/" + self.path + "_weight.h5")

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
		with open("../NNPostProcessing/Assets/Resources/Model/" + self.path + "_" + self.dataset_name + ".json", "w") as json_file:
		    json_file.write(formatstr)
		print(self.loaded_model.summary())
		plot_model(self.loaded_model, to_file='model/model_architecture.png', show_shapes=True)

if __name__ == '__main__':
	Exporter()