import os
import pandas as pd
import numpy as np
import cv2

import string
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class Dataset():
    def __init__(self, df,  path, max_pad,
                 tokenizer, input_size = (224,224,3)):
        self.image1 = df.Image1
        self.image2 = df.Image2
        
        self.number = df.Number
        self.path = path
        self.input_size = input_size
        self.pad = max_pad
        
        # preprocessing on the text
        # self.caption = df[caption_col].apply(lambda x: text_preprocess(x))
        self.tokenizer = tokenizer
        self.caption = df.Caption
    
        
    def __getitem__(self, i):
        image1 = cv2.imread(os.path.join(self.path, self.image1.iloc[i].lstrip()), cv2.IMREAD_UNCHANGED)/255
        image1 = cv2.resize(image1, (self.input_size[0], self.input_size[1]), interpolation = cv2.INTER_NEAREST)
        image1 = np.array(image1).reshape(1,self.input_size[0],self.input_size[1],self.input_size[2])
        
        image2 = cv2.imread(os.path.join(self.path, self.image2.iloc[i].lstrip()), cv2.IMREAD_UNCHANGED)/255
        image2 = cv2.resize(image2, (self.input_size[0], self.input_size[1]), interpolation = cv2.INTER_NEAREST)
        image2 = np.array(image2).reshape(1,self.input_size[0],self.input_size[1],self.input_size[2])
        
        # return the caption value
        caption = self.tokenizer.texts_to_sequences(self.caption.iloc[i:i+1])
        caption = pad_sequences(caption, maxlen = self.pad, padding = 'post')
        
        
        return image1, image2, caption

    def __len__(self):
        return len(self.image1)
    
    def text_preprocess(self, capt):
        capt = capt.split()

        capt = [word.lower() for word in capt]

        # remove punctuations from the string; like $#&
        table = str.maketrans('', '', string.punctuation)
        capt = [word.translate(table) for word in capt]

        # removing the hanging letters like s or t
        capt = [word for word in capt if len(word)>1] 
        capt = [word for word in capt if word not in STOP_WORDS]

        return ' '.join(capt)