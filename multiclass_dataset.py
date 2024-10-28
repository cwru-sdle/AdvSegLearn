import torch
import pandas as pd
from skimage.io import imread
import numpy as np
import random
from torchvision import tv_tensors # torch.clamp and tv_tensors is needed to use v2 transforms
class Multiclass_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x_sup: list[list[str]] = [],
        y: list[list[str]] = [],
        x_unsup: list[list[str]] = [],
        initalization_transform=None,
        getitem_transform=None,
        imgs_per_transform:int=1,
    ):
        if(len(x_sup)!=0):
            self.check_list_list_str(x_sup,'x_sup')
        if(len(y)!=0):
            self.check_list_list_str(y,'y')
        if(len(x_unsup)!=0):
            self.check_list_list_str(x_unsup,'x_unsup')
        self.initalization_transform = initalization_transform
        self.getitem_transform = getitem_transform
        self.imgs_per_transform = imgs_per_transform
        self.data = self.initialize_data(x_sup, y, x_unsup)
        print(f'Initalization Complete: {len(self.data)}')
    @staticmethod
    def check_list_list_str(input,name):
        if not isinstance(input, list):
            raise ValueError(name+'must be a list was a '+str(type(input)))
        if not isinstance(input[0], list):
            raise ValueError(name+'[0] must be a list was a '+str(type(input[0])))
        if not isinstance(input[0][0], str):
            raise ValueError(name+'[0][0] must be a str was a '+str(type(input[0][0])))
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            print(f'Cuda available. {torch.cuda.device_count()} devices available.')
            return torch.device('cuda', 0)
        else:
            print('Cuda unavailable.')
            return torch.device('cpu')
    @staticmethod
    def imgs_to_channel(paths: list[str], printing=False):
        imgs = []
        for path in paths:
            try:
                try:
                    img = imread(path, as_gray=True)
                except ValueError:
                    img = imread(path)
                    if img.ndim ==3 and img.shape[2]==2:
                        img = img[:,:,1]
                    else:
                        raise ValueError(f'Image has unusual dimensions: {img.shape}\nImage is at: {path}')
                imgs.append(img)
                if printing:
                    df_describe = pd.DataFrame(img)
                    print(df_describe.describe())
                    print(path)
            except OSError as e:
                print(f'Error: Invalid file path. \nStandard Error message: {e}')
        try:
            output = np.stack(imgs, axis=0)
        except ValueError:
            print('Path caused ValueError'+paths[0][-30:0])
        output = torch.from_numpy(output.astype(np.float32))            
        return output
    @staticmethod
    def preprocessing(input):
        return input
    def validate_supervised_data(self, x_sup, y):
        for x_i in range(len(x_sup) - 1):
            if len(x_sup[x_i]) != len(x_sup[x_i + 1]):
                raise IndexError('The number of items in each input list do not match')
        for y_i in range(len(y) - 1):
            if len(y[y_i]) != len(y[y_i + 1]):
                raise IndexError('The number of items in each output list do not match')
        if len(x_sup[0]) != len(y[0]):
            raise IndexError('The number of elements in each input and output lists do not match'+f'\nx_sup[0]: '+str(len(x_sup[0]))+f'\ny_sup[0]: '+str(len(y[0])))

    def initialize_data(self, x_sup, y, x_unsup):
        data = []
        if (x_sup or y):
            self.validate_supervised_data(x_sup, y)
            data.extend(self.create_supervised_data(x_sup, y))
        if x_unsup:
            data.extend(self.create_unsupervised_data(x_unsup))
        if not data:
            raise IndexError('There is no data. The supervised and unsupervised list is empty')
        return data
    @staticmethod
    def convert_1(input:torch.tensor):
        if(torch.max(input).item()>((2**16)-1)):
            output - input / ((2**32) -1)
        elif(torch.max(input).item()>((2**8)-1)):
            output = input/((2**16)-1)
        elif(torch.max(input).item()>1):
            output = input/((2**8)-1)
        else:
            output = input
        if(torch.max(output).item()>1):
            raise ValueError('Tensor should have max of 1, but was: '+str(torch.max(output)))
        return output

    def create_supervised_data(self, x_sup, y):
        data = []
        for x_i in range(len(x_sup[0])):
            temp_x = [x_sup[x_j][x_i] for x_j in range(len(x_sup))]
            x_instance = self.imgs_to_channel(temp_x)
            temp_y = [y[y_j][x_i] for y_j in range(len(y))]
            y_instance = self.imgs_to_channel(temp_y)
            if callable(self.initalization_transform):
                for i in range(0,self.imgs_per_transform):
                    x_instance = self.convert_1(x_instance)
                    y_instance = self.convert_1(y_instance)
                    x_instance = tv_tensors.Image(x_instance)
                    y_instance = tv_tensors.Image(y_instance)
                    image, mask= self.initalization_transform(x_instance,y_instance)
                    image = torch.clamp(image,0,1-0.00001)
                    mask = torch.clamp(mask,0,1-0.00001)
                    data.append((image, mask))
            elif isinstance(self.initalization_transform,list) and callable(self.initalization_transform[0]):
                for i in range(0,self.imgs_per_transform):
                    for initalization_transform in self.initalization_transform:
                        x_instance = self.convert_1(x_instance)
                        y_instance = self.convert_1(y_instance)
                        x_instance = tv_tensors.Image(x_instance)
                        y_instance = tv_tensors.Image(y_instance)
                        image, mask= initalization_transform(x_instance,y_instance)
                        image = torch.clamp(image,0,1-0.00001)
                        mask = torch.clamp(mask,0,1-0.00001)
                        # print('x: '+str(torch.max(x_instance))+str(torch.min(x_instance)))
                        # print('y: '+str(torch.max(y_instance))+str(torch.min(y_instance)))
                        data.append((image, mask))
            else:
                data.append((x_instance, y_instance))
        return data

    def create_unsupervised_data(self, x_unsup):
        data = []
        for x_i in range(len(x_unsup[0])):
            temp = [x_unsup[x_j][x_i] for x_j in range(len(x_unsup))]
            instance = self.imgs_to_channel(temp)

            if callable(self.initalization_transform):
                for i in range(0,self.imgs_per_transform):
                    x_instance = self.convert_1(instance)
                    x_instance = tv_tensors.Image(x_instance)
                    image= self.initalization_transform(x_instance)
                    image = torch.clamp(image,0,1-0.00001)
                    data.append((image,))
            elif isinstance(self.initalization_transform,list) and callable(self.initalization_transform[0]):
                for i in range(0,self.imgs_per_transform):
                    for initalization_transform in self.initalization_transform:
                        x_instance = self.convert_1(instance)
                        x_instance = tv_tensors.Image(x_instance)
                        image,= initalization_transform(x_instance)
                        image = torch.clamp(image,0,1-0.00001)
                        data.append((image,))
            else:
                data.append((instance))
        return data

    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else 0

    def __getitem__(self, idx):
        if len(self.data[idx]) == 2:
            if callable(self.getitem_transform):
                x_instance = self.convert_1(self.data[idx][0])
                y_instance = self.convert_1(self.data[idx][1])
                x_instance = tv_tensors.Image(x_instance)
                y_instance = tv_tensors.Image(y_instance)
                image, mask= self.getitem_transform(x_instance,y_instance)
                image = torch.clamp(image,0,1-0.0001)
                mask = torch.clamp(mask,0,1-0.0001)
                return image, mask
            elif self.getitem_transform is None:
                return self.data[idx][0], self.data[idx][1]
            else:
                raise ValueError('Type of getitem_transform should be a function or none. It is: '+str(type(self.getitem_transform)))
            
        elif len(self.data[idx]) == 1:
            if callable(self.getitem_transform):
                x_instance = self.convert_1(self.data[idx][0])
                x_instance = tv_tensors.Image(x_instance)
                image= self.getitem_transform(x_instance)
                image = torch.clamp(image,0,1-0.0001)
                return image
            elif self.getitem_transform is None:
                return self.data[idx][0]
            else:
                raise ValueError('Type of getitem_transform should be a function or none. It is: '+str(type(self.getitem_transform)))
        else:
            print(self.data[idx])
            print(len(self.data[idx]))
            print(len(len(self.data[idx])))
            raise Exception(f'Datapoint {idx} is not a tuple with 1 or 2 elements.')