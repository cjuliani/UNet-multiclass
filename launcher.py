import os,json
os.chdir("C:/Users/cyrilj/Desktop/github-unet-multiclass")
import numpy as np
import datasets,unet

if __name__ == '__main__':
	np.random.seed(92457)
	
# import pre-defined parameters
with open('config.json') as file:
	params = json.load(file)
	
# get paths for datasets
data_path = params['data_path']
valid_path = params['valid_path']
test_path = params['test_path']

in_size = params['img_size']									# input size
in_dim = params['input_dimension']								# input dimension
n_class = params['classes']										# number of classes
val_pct = params['validation_percent']							# percent of training set for validation | use 0 if pre-defined validation set

print('\nParameters')
print('input size : {}x{}x{}'.format(in_size,in_size,in_dim))
print('classes :', n_class)
print('validation (random selection) : '+str(val_pct)+'%')

# get training and validation sets
img_path = os.path.join(data_path, 'images')
anno_path = os.path.join(data_path, 'annotations')
training_data, validation_data = datasets.read_data_sets(img_path, anno_path, in_size, val_pct)

if val_pct <= 0:
	# get pre-defined validation set
	img_names = datasets.load_train(path=valid_path+'images')			# input images
	anno_names = datasets.load_train(path=valid_path+'annotations')		# target annotations (binary masks)
	validation_data = datasets.generator(img_names, anno_names, in_size)
	print("Validation examples (manual selection): ", validation_data._num_examples)

# build the model
model = unet.unet()

model.train(training_data=training_data,
	validation_data=validation_data)

	img_names = datasets.load_train(path=test_path+'images')            # input images
	anno_names = datasets.load_train(path=test_path+'annotations')      # annotations (binary masks)
	test_data = datasets.generator(img_names, anno_names, in_size)    	# get dataset

	model.test(test_data=test_data,restore=True)

	model.predict(in_size=512,output=1)