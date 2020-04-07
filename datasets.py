from __future__ import print_function
from __future__ import division
import os,random,util,json
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
from itertools import groupby 
from operator import itemgetter
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

os.chdir("C:/Users/cyrilj/Desktop/github-unet-multiclass/")

# import pre-defined parameters
with open('config.json') as file:
    params = json.load(file)

# get paths for datasets
data_path = params['data_path']
img_path = os.path.join(data_path, 'images')
anno_path = os.path.join(data_path, 'annotations')

def get_validation_data(ft,lb,val_pct):
    '''
    Dispatch data into training and validation sets
    '''
    # Shuffle
    nb_examples = len(ft)
    idx_d = list(range(nb_examples))
    np.random.shuffle(idx_d)
    
    # Threshold for test values
    valid_trsh = int((nb_examples * val_pct) / 100)
    
    # Re-organize
    train_f = [ ft[i] for i in idx_d[valid_trsh:] ]
    train_lb = [ lb[i] for i in idx_d[valid_trsh:] ]
    val_f = [ ft[i] for i in idx_d[0:valid_trsh] ]
    val_lb = [ lb[i] for i in idx_d[0:valid_trsh] ]
    
    return train_f, train_lb, val_f, val_lb

def load_train(path):
    '''
    Provides training from existing sets (directories)
    '''
    # Collect existing folders
    dir = []
    for x in os.walk(path):
        try:
            dir.append(x[0].split('\\')[1])
        except:
            pass

    # Collect png files in folders 
    files_in_folders = []
    for folder in dir:
        folder_path = os.path.join(path, folder)

        # Collect data names in current folder
        files = []
        for file in os.walk(folder_path):
            files.append(file)
            files = [os.path.join(folder_path, name) for name in files[0][2]]
            
            # Rule out non-PNG files
            cnt = 0
            for f in files:
                formt = f.split('\\')[-1].split('.')[-1]
                if formt == 'png' or formt == 'PNG':
                    pass
                else:
                    files.pop(cnt)
                cnt += 1

        # Group images given annotation index
        ints = [ int(name.split('_')[-1].split('.')[0]) for name in files]
        ints = [ [idx,int_] for idx,int_ in zip(range(len(ints)),ints) ]
        idxs = sorted(ints, key=itemgetter(1))
        idxs = np.array(idxs)[:,0]
        files_ = [files[i] for i in idxs]
        files_ = [list(i) for j, i in groupby(files_, lambda a: a.split('_')[-1].split('.')[0])]
        files_in_folders.append(files_)

    return [e for sub in files_in_folders for e in sub]

class generator(object):
    '''
    Dataset object sorting train images 
    plus related annotations, and manages batching
    '''
    def __init__(self, img_names, anno_names, img_size):
        self._num_examples = len(img_names)
        self._img_names = img_names
        self._anno_names = anno_names
        self._epochs_done = 0
        self._index_in_epoch = 0
        self._img_size = img_size
        self._indexes = list(range(len(img_names)))
        #
        self.n = 0              # rotation angle
        self.v = 0              # translation rate
        self.scl = 1            # scale ratio
        self.noise = 0
        self.blur = 0
        self.contrast = 1
        self.brightness = 0     
        #
        self.flip = 0
        self.deform = 0
        self.light = 0
        
    def rotate(self, arr):
        return ndimage.rotate(input= arr,angle= self.n,reshape=False)
    
    def clipped_zoom(self, img, zoom_factor, **kwargs):
        '''
        Data scaling
        source: https://stackoverflow.com/questions/37119071/
        scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        '''
        h, w = img.shape[:2]
        # For multichannel images we don't want to apply the zoom factor to the RGB
        # dimension, so instead we create a tuple of zoom factors, one per array
        # dimension, with 1's for any trailing dimensions after the width and height.
        zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
        
        # Zooming out
        if zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            #
            # Zero-padding
            out = np.zeros_like(img)
            out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

        # Zooming in
        elif zoom_factor > 1:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            #
            out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]
        # If zoom_factor == 1, just return the input array
        else:
            out = img
        return out

    def scale(self, arr):
        arr = ndimage.rotate(input= arr,angle= self.n,reshape=False)
        return self.clipped_zoom(arr, self.scl)
    
    def translate(self, arr):
        v_ = int( (self._img_size/100) * self.v )
        for c in range(arr.shape[-1]):
            arr_TMP = ndimage.shift(arr[:,:,c], v_)
            arr[:,:,c] = arr_TMP
        return arr
    
    def flip_h(self, arr):
        return np.fliplr(arr)
    
    def flip_v(self, arr):
        return np.flipud(arr)
    
    def no_flip(self, arr):
        return arr

    def change_contrast(self, arr):
        factor = float(self.contrast)
        arr = (arr*255).astype(np.uint8)
        arr = np.clip(128 + factor * arr - factor * 128, 0, 255).astype(np.uint8)
        return arr / 255
    
    def change_brightness(self,arr):
        arr = np.int16(arr*255)
        arr = (arr + self.brightness)
        arr = np.clip(arr,0,255).astype(np.uint8)
        return arr / 255
    
    def gauss_noise(self, arr):
        gauss = np.random.normal(0,self.noise,(self._img_size,self._img_size))
        for c in range(arr.shape[-1]):
            arr[:,:,c] = np.add(arr[:,:,c], gauss)
        return arr
    
    def gauss_blur(self, arr):
        return gaussian_filter(arr,self.blur)
        
    def augment_geometry(self, arr):
        # select flipping or non-flipping mode
        functions = [ self.flip_h, self.flip_v, self.no_flip ]
        arr = functions[self.flip](arr)
        #
        arr =  self.rotate(arr)
        functions = [ self.scale, self.translate ]
        return functions[self.deform](arr)
    
    def augment_magnitude(self, arr):
        # select mode
        functions = [ self.change_brightness, self.change_contrast, self.gauss_blur, self.gauss_noise ]
        return functions[self.light](arr)
    
    def next_batch(self, batch_size, data_aug=True):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        # Randomize examples after every epoch
        # i.e., after nb of indexes exceed total nb of examples
        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
            #
            np.random.shuffle(self._indexes)
            self._img_names = [self._img_names[i] for i in self._indexes]
            self._anno_names = [self._anno_names[i] for i in self._indexes]
        end = self._index_in_epoch
        
        # Lists of images and masks
        images, annos = [],[]

        # Iterate through the batch start-end to get img + annotation
        # and augment data for every increment of batch
        for i in range(start, end):
            
            # Activate data augmentation if True
            if data_aug is True:
                # Pick up random parameter
                self.v = random.choice(np.arange(-50,50,5))
                self.scl = random.choice([0.6,0.8,1.0,1.2,1.5])
                self.n = random.choice(np.arange(0,360,20))
                self.noise = random.choice(np.arange(0,0.15,0.02))
                self.blur = random.choice(np.arange(0,1.3,0.1))
                self.contrast = random.choice(np.arange(1,2.5,0.1))
                self.brightness = random.choice(np.arange(-100,100,20))
                self.gamma = random.choice(np.arange(0,50,10))
                #
                self.flip = random.choice(np.arange(0,3))
                self.deform = random.choice(np.arange(0,2))
                self.light = random.choice(np.arange(0,4))
            
            # Images
            img_name = self._img_names[i]
            img_0 = []
            for img_ in img_name:
                img = util.remove_transparency(Image.open(img_))
                img = img.resize( (self._img_size,self._img_size) )
                img = np.array(img)
                img = img.astype(np.float32)
                # normalize RGB by 255, otherwise normalize data by min-max given that no-data values are 0s
                img = np.multiply(img, 1.0 / 255.0) if len(img.shape) == 3 else ( (img - np.min(img[img!=0])) / (np.max(img)-np.min(img[img!=0])) ).clip(min=0)
                img = np.reshape( img, (self._img_size*self._img_size, -1) )
                img_0.append(img)
            image = np.concatenate( img_0, axis=-1)
            image = np.reshape(image, (self._img_size, self._img_size, image.shape[-1]) )
            image = self.augment_magnitude(image) if data_aug is True else image
            image = self.augment_geometry(image) if data_aug is True else image
            images.append(image)
            
            # Annotations
            anno_names_ = self._anno_names[i]
            anno_l = []
            anno_0 = np.zeros(( self._img_size, self._img_size))  # final mask taking maximum class integer
            for nm in anno_names_:
                anno = util.remove_transparency(Image.open(nm))
                anno = anno.resize( (self._img_size,self._img_size) )
                anno = anno.convert('L')
                anno = np.array(anno.point(lambda x: 0 if x>128 else 1),dtype='int32')
                anno = self.augment_geometry( np.expand_dims(anno,axis=-1) ) if data_aug is True else np.expand_dims(anno,axis=-1)
                anno_l.append(anno[...,-1])
            annos.append(anno_l)
            
        # Transform lists into arrays
        images = np.array(images)
        annos = np.array(annos)
        annos = np.moveaxis(annos, 1, -1)   # last axis must be the input dimension
        
        return images, annos, self._img_names[start:end], self._anno_names[start:end]

def read_data_sets(img_path,anno_path,img_size,val_pct):
    '''
    Read datasets - load data and split (randomly) these data into validation and training sets
    '''
    img_names = load_train(path=img_path)
    anno_names = load_train(path=anno_path)
    try:
        img_train, anno_train, img_valid, anno_valid = get_validation_data(ft=img_names,lb=anno_names,val_pct=val_pct)
    except IndexError:
        print('The size of datasets for inputs and annotations are different.')
    
    # Shuffle loaded img + annotations
    print("\nTraining examples: ", len(img_train))
    print("Annotations per input: ", len(anno_names[0]))
    if val_pct <= 0:
        print("Validation examples (random selection): ", len(img_valid))
    
    # Fill in data to new object
    training_set = generator(img_train, anno_train, img_size)
    validation_set = generator(img_valid, anno_valid, img_size)
    
    return training_set,validation_set