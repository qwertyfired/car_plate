from tqdm import tqdm
import torch.optim
from torch.utils.data import DataLoader , random_split
import torch
from torchvision import transforms
import cv2
import numpy as np
from model.vgg16 import VGG16
import matplotlib.pyplot as plt

from data_loader import ImageDataset
from VGG_FCN import VGGFCN

# VGG_types = {
#         'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#         'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512 ,'M', 512, 512, 'M'],
#         'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#         'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#         }

# scale = (256, 256)
# transform = transforms.Compose(
#     [
#         transforms.Grayscale(),
#         transforms.Resize(scale),
#         transforms.ToTensor(),
#     ]
# )

dataset = ImageDataset(data_path= 'test_hdf5.hdf5', scaling= True)
m=len(dataset)
train_data, test_data = random_split(dataset, [int(m-10), int(10)])

dataloader = DataLoader(dataset=train_data, batch_size=20, shuffle=True, drop_last=False)
testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, drop_last=False)

# testset = LpDataset(path='./data', train=False, transform=transform)
# testloader = DataLoader(dataset=testset, batch_size=1, shuffle=True, drop_last=False)
print("data success")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = VGGFCN(model='VGG16', in_channels=1, num_output=4, init_weight=True)
model = VGG16(1)
model.to(device)
print("model success")

# Define model optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# optimizer= torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = torch.nn.MSELoss().to(device)

epoches = 30
loss_list=[]

model.train()
for m in range(epoches):
    all_loss = 0
    for i in tqdm(dataloader):
        img, label= i
        img= img.to(device)
        label= label.to(device)
        # size= size.to(device)
        optimizer.zero_grad()

        pred_label = model(img)
        # pred_label = torch.squeeze(pred_label)
        # pred_label= torch.reshape(pred_label, (-1, 4))
        label = torch.unsqueeze(label,-1)
        label = torch.unsqueeze(label,-1)
        loss = loss_fn(pred_label, label)

        all_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'epoch: {m}')
    all_loss = all_loss/len(dataloader)
    print(f'loss : {all_loss}')
    loss_list.append(all_loss)
x= np.arange(30)
plt.plot(x, loss_list)
plt.title('epoch: 30, data:2000')
plt.xlabel("epoches")
plt.ylabel("accuracy")
plt.savefig('images_results_2000/size2000.png')
'''
model.eval()
for batch_idx, samples in enumerate(testloader):
    if(batch_idx > 10):
        break
    img, label= samples
    img= img.to(device)
    label= label.to(device)
    # width = size['width']
    # height = size['height']
    # print(label)
    pred = model(img)
    pred = pred.detach().cpu().numpy()
    print(f'find {pred[0][0]} {pred[0][1]} {pred[0][2]} {pred[0][3]}')
    img = torch.squeeze(img,0)
    img = img.permute(1,2,0)


    img = img.detach().cpu().numpy()
    img = (img * 255.).astype(np.uint8).copy()

    top_left_x = int(pred[0][0]*255)    
    top_left_y = int(pred[0][1]*255)
    bottom_right_x = int(pred[0][2]*255)
    bottom_right_y = int(pred[0][3]*255)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    img = cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0,255,0), 2)
    cv2.imwrite(f'images_results/{batch_idx}.png', img[:,:,::-1])
    print(f"complete {top_left_x},{top_left_y},{bottom_right_x},{bottom_right_y}")
    '''
