from keras.models import model_from_json, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from data_loader import DataLoader
import json
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='terrain', help='name of the dataset')
parser.add_argument('--modelpath', dest='modelpath', default='model', help='name of model')
args = parser.parse_args()

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
		imgs_A, imgs_B = data_loader.load_data(batch_size=1, is_testing=True)
		interresult0 = imgs_B.tolist()
		fake_A = self.loaded_model.predict(imgs_B)
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
	Debugger(path = args.modelpath, dataset_name = args.dataset_name)