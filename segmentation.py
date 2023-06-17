import cv2
import numpy as np
import imutils
import torch
from glob import glob
from utils.functions import *
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fontpath = "./font/SCDream4.otf"
font = ImageFont.truetype(fontpath, 15)

image_path = ''

for num, img in enumerate(tqdm(glob(image_path))):
	image = cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# gray = imutils.resize(gray, width=250, inter=cv2.INTER_LINEAR)
	gray = cv2.resize(gray, (250,60))
	image_y, image_x= gray.shape[:2]
	image =gray.copy()
	final_image =cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	gray_mean=np.mean(gray)
	gray[gray< gray_mean*0.5] =0
	gray[gray> gray_mean*0.7] =255
	thresh = np.where(gray>=gray_mean,0,255).astype(np.uint8)

	_, labels = cv2.connectedComponents(thresh)
	mask = np.zeros(thresh.shape, dtype="uint8")

	thresh_list =list()
	thresh_idx =list()
	for (i, label) in enumerate(np.unique(labels)):
		# If this is the background label, ignore it
		if label == 0:
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if  numPixels>70 and numPixels <1500:
			mask = cv2.add(mask, labelMask)

	(contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Sort the contours by their x-axis position, ensuring that we read the numbers from left to right
	contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

	####test
	contour_y = list()
	contour_w = list()
	contour_h = list()
	area =list()
	pre_x, pre_y, pre_w, pre_h = 0,0,0,0
	final_contour=list()
	for (contour, _) in contours:
		# Compute the bounding box for the rectangle
		(x, y, w, h) = cv2.boundingRect(contour)
		if h >3  and w/h<5  and x+w<=image_x: #and h/w<6 마지막으로 변경
			if (x<12 or x>230) and (x+w>=image_x or y+h >=image_y): continue 
			if (x<15 or x>180) and (abs(w-h)<=8 or h<15) and w+h<52: continue #x+w>=image_x-1 or
			if (x<15  and w<16 and h<35): continue #and abs(w-h)<=10
			if (w>39): continue
			if (x>200 and w<15 and h<35): continue
			if (x<12 and y+h>=image_y): continue
			if (x>210 and h/w>5): continue
			if (y<5 and abs(w-h)<2): continue
			if (x>210 and y<5 and x+w>=image_x): continue
			if (x>210 and x+w>=image_x and y+h >=image_y): continue
			if ((x< 10 or x >220) and (h <=10 or y+h == image_y)): continue
			if (x <5 and h>50): continue
			if (x <5 and h/w>6): continue
			if (x <5 and abs(w-h)<10): continue
			if (x <5 and h+w<51): continue
			if (x <20 and h+w<45): continue
			if overlap(pre_x,pre_w, x,w):
				x,y,w,h = combination(pre_x,pre_y,pre_w,pre_h, x,y,w,h)
				del contour_w[-1]
				del final_contour[-1]
			contour_w.append(w)
			final_contour.append([x,y,w,h])
			pre_x, pre_y, pre_w, pre_h = x,y,w,h
	w_mean = np.mean(contour_w)
	kor =list()
	'''
	이상 발생 -> 글자가 2개로 분리 (||) 혹은 (=) 형식으로. 비율을 확인한다
	'''
	
	if len(final_contour) not in [7,8,9]:
		nodetection +=1
		no_list.append(num)
	final_bbox =list()
	final_char=list()
	if len(final_contour) not in [8,9]:
		# print("number is 7")
		sum +=1

		for idx,contour in enumerate(final_contour):
			(x, y, w, h) = contour
			if w > 10 or h > 10:
				# Crop the ROI and then threshold the greyscale ROI to reveal the digit
				
				final_bbox.append([x,y-1,w,h])


	elif len(final_contour) == 8:
		# print("number is 8")
		previous_number=0
		previous_sum=0
		for idx,contour in enumerate(final_contour):
			(x, y, w, h) = contour
			if idx not in [2,3]:
				# Crop the ROI and then threshold the greyscale ROI to reveal the digit
				
				final_bbox.append([x,y,w,h])

				if idx==1: 
					previous_sum = (w ** 2 + h ** 2) ** 0.5
			else :
				if (w ** 2 + h ** 2) ** 0.5 > previous_sum * 0.9 and previous_number==0:
					
					final_bbox.append([x,y,w,h])
					previous_number=1
	
				else:
					kor.append([x,y,w,h])
				
		if len(kor) ==1:
			(x,y,w,h) = kor[0]
			if(y>0): y=y-1
			roi = image[y:y + h, x:x + w+1]
			
			final_bbox.append([x,y-1,w,h])
			kor.clear()
	elif len(final_contour) == 9:
		sum +=1
		for idx,contour in enumerate(final_contour):
			(x, y, w, h) = contour
		
			if idx not in [3,4]:
				
				final_bbox.append([x,y,w,h])
				
			else:
				kor.append([x,y,w,h])
	if len(kor) >1:
		if (abs(kor[0][0] - kor[1][0])<=8 and kor[0][1] > kor[1][1]):   #(abs(kor[0][0] - kor[1][0])<=5 and kor[0][1] > kor[1][1]): 
			y=kor[1][1]
			y_h=kor[0][1]+kor[0][3]
			x=kor[1][0]
			x_w=kor[0][0]+kor[0][2]
		else:
			y=kor[0][1]
			y_h=kor[1][1]+kor[1][3]
			x=kor[0][0]
			x_w=kor[1][0]+kor[1][2]
		h= y_h -y
		w = x_w-x
		if(y>0): y=y-1
		if (x+w<249): w+=1
		if w > 60:
			nodetection +=1
			continue
		
		final_bbox.append([x,y-1,w,h])
		
	im_pil = Image.fromarray(final_image)
	draw = ImageDraw.Draw(im_pil)
	for i in range(len(final_bbox)):
		box_location = final_bbox[i]
		x,y,w,h = box_location
		draw.rectangle(xy=(x,y,x+w,y+h), outline=(0,255,0))
		
	im_pil.save(f"./bounding_{num}.png","PNG")
print(nodetection)



    
