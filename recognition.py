import json
import os
from pickle import TRUE
import torch
import random
import numpy as np
import cv2
import os
import math
from torchvision import transforms
char = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9',
          '10':'가', '11':'나', '12':'다', '13':'라', '14':'마', '15':'거', '16':'너', '17':'더', '18':'러', '19':'머', '20':'버', '21':'서',
                   '22':'어', '23':'저', '24':'고', '25':'노', '26':'도', '27':'로', '28':'모', '29':'보', '30':'소', '31':'오', '32':'조', '33':'구', '34':'누',
                     '35':'두', '36':'루', '37':'무', '38':'부', '39':'수', '40':'우', '41':'주', '42':'하', '43':'허', '44':'호'}



def total_detection(img, model):
    global minus_data
    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    
    try:
        crop = np.where(gray>gray_mean,0,255).astype(np.uint8)
    except:
        return -2
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (64,64))
  
    crop = crop[np.newaxis, np.newaxis, :, :]
    crop = torch.tensor(crop, dtype=torch.float32)/ 255.
    crop = crop.to(device)
    
    pred = model(crop)
    pred = pred.detach().cpu().numpy()
    pred = pred.argmax(1)
    pred = char[str(pred[0])]
    # print(pred)
    return pred
