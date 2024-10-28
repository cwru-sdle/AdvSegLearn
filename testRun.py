import os
import pandas as pd
fractography_path = '/mnt/vstor/CSE_MSE_RXF131/staging/mds3/fractography'
manuel_mask_path = '/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/keyence-fractography/manuel_mask'
#Adds File if Condition(path) return True
def recursive_search(condition,path:str, file_list:list):
    if os.path.isdir(path):
        for path_loop in os.listdir(path):
            recursive_search(condition,os.path.join(path,path_loop),file_list)
    else:
        if(condition(path)):
            file_list.append(path)
    return file_list

#Definition Different columns that would be valuable to have
def stitched(path):
    if 'stitched' in path.lower():
        return True
    elif'composite' in path.lower():
        return True
    else:
        return False

def exists(path):
    return True 
def png(path):
    if ('.png' in path.lower()):
        return True
    else:
        return False
def marked(path):
    if 'marked' in path.lower():
        return True
    else:
        return False

def initiation(path):
    if '_001' in path:
        return True
    elif 'initiation' in path.lower():
        return True
    else:
        return False
def marked_and_initiation(path):
    if initiation(path) and marked(path):
        return True
    else:
        return False
def marked_and_stitched(path):
    if marked(path) and stitched(path):
        return True
    else:
        return False
def fatigue(path):
    if 'fatigue' in path.lower():
        return True
    else:
        return False
def overload(path):
    if 'overload' in path.lower():
        return True
    else:
        return False
def fatigue_and_png(path):
    if png(path) and fatigue(path):
        return True
    else:
        return False
def stitched_and_png_not_fatigue(path):
    if png(path) and stitched(path) and not fatigue(path):
        return True
    else:
        return False
path_list = [fractography_path,manuel_mask_path]
condition_list = [exists,stitched_and_png_not_fatigue,initiation,marked_and_initiation, marked_and_stitched,fatigue,overload,fatigue_and_png]
name = []
column_dict = {}
i=0
for top_folder in path_list:
    for column in condition_list:
        temp_list = []
        column_dict[column.__name__+'_'+os.path.basename(top_folder)]=recursive_search(
            column,
            top_folder,
            temp_list
        )
        name.append((column.__name__+'_'+os.path.basename(top_folder),len(column_dict[column.__name__+'_'+os.path.basename(top_folder)])))
        print(str(name[i]) +f'\tPosition: {i}')
        i+=1

import re
check = re.compile(r'''
    ^(EP|NASA|CMU)  # Start with EP, NASA, or CMU (case insensitive)
    [-_]?           # Optional separator
    (\d+|O\d+)      # Number or O followed by number
    .*?             # Any characters in between (non-greedy)
    (?:             # Non-capturing group
        [-_]?V?     # Optional separator and V
        ([E\d]+)       # Version number
        (?:[-_](\d+))?  # Optional additional number
    )?
    .*?             # Any characters in between (non-greedy)
    \.(png|tif)$    # File extension (png or tif)
''', re.VERBOSE | re.IGNORECASE)
def output(pattern):
    match = re.search(check,pattern)
    if(match):
        # print(match.groups())
        # print(match)
        return match.group(1).lower()+match.group(2), match.group(3).lower() ,match.group(4) if match.group(4) else None
    else:
        return None

#Testing Function on Data
# for key in column_dict:
#     for field in column_dict[key]:
#         if not output(field.split('/')[-1]):
#             print(field.split('/')[-1])

            
dataframe_list = []
for i, key in enumerate(column_dict):
    type_column = []
    series_column = []
    posit_idx_column = []
    basename = []
    for j, field in enumerate(column_dict[key]):
        if output(field.split('/')[-1]):
            type_inst, series_inst, posit_idx_inst = output(field.split('/')[-1])
            type_column.append(type_inst)
            series_column.append(series_inst)
            posit_idx_column.append(posit_idx_inst)
            basename.append(field.split('/')[-1])
        else:
            type_inst=None
            series_inst=None
            posit_idx_inst=None
    path_column = pd.Series(column_dict[key],name='path')
    type_column = pd.Series(type_column, name='type')
    series_column = pd.Series(series_column,name='series')
    posit_idx_column = pd.Series(posit_idx_column,name='posit_idx')
    basename_column = pd.Series(basename,name='basename')
    dataframe_list.append(
        pd.concat(
            [
                path_column,
                type_column,
                series_column,
                posit_idx_column,
                basename_column
            ],
            axis=1
        )
    )
    print(str(name[i])+ '\tPosition: '+str(i))
    print(dataframe_list[i].head(7))


import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from unet import Unet
from discriminator_loss import discriminator_loss
from semi_supervised_loss import semi_supervised_loss
from multiclass_dataset import Multiclass_dataset
from train_GAN import train_GAN
from make_dataset import dataset_setup

import torch
# Initalize The Dataset
merge = pd.merge(dataframe_list[-1],dataframe_list[9],on=['type','series','posit_idx'],suffixes=['_mask','_stitched'])
print(len(merge))
print(len(dataframe_list[-1]))
# for i in range(len(merge)):
#     print(merge['basename_stitched'][i]+str(merge['posit_idx'][i]) +'\t'+str(merge['series'][i])+'\t'+str(merge['type'][i]))
disunion = dataframe_list[9]['path']
train_dl, valid_dl = dataset_setup(
    [merge['path_stitched']],
    [merge['path_mask']],
    .80,
    .15,
    .05,
)
def size_transform(size:list[int]):
    if len(size)!=2:
        raise ValueError('The size must a list of 2 integers')
    def apply_transform(image):
        return cv2.resize(image,size)

resize_func = size_transform([512,512])
unsegmented_raw = Multiclass_dataset(x_unsup=[disunion.dropna()],transform=resize_func)
# x_unsup is the 
segmentor = Unet(
    input_channels = 1,
    output_channels=1,
    encoder_pairs=4,
    initial_features=64,
    features_expanded=2,
)
discriminator = Unet(
    input_channels = 1,
    output_channels =1,
)

train_GAN(
    discriminator_loss(),
    torch.nn.BCELoss(),
    semi_supervised_loss(),
    segmentor,
    discriminator,
    train_dl,
    valid_dl,
    unsegmented_raw,
    epochs=5,
    accumulation_steps=1,
    save_path = None,
    learning_rate=0.0001,
)