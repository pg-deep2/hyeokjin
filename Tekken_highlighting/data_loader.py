from __future__ import print_function
import os, random
import cv2
import numpy as np
import torch, time
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import functools

class RawVideo_Dataset(torch.utils.data.Dataset):
    #dataset을 가져와 선처리
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot
        videolist = os.listdir(dataroot)
        self.videolist = [os.path.join(self.dataroot, video) for video in videolist]
        self.transforms = transform if transform is not None else lambda x: x

    #동영상에서 한 개의 프레임 반환.
    def __getitem__(self, index):
        vidcap = cv2.VideoCapture(self.videolist[index])
        frames = []
        h_label = self.videolist[index].split("\\")[-2]


        # cv2를 이용해 video reading!!
        f=0
        while (vidcap.isOpened()): #isOpened() : 비디오 캡쳐가 된 경우 true 반환.
            ret, frame = vidcap.read() #caption후 다음 frame을 return.

            if ret:
                # (height,width,channel) -> (channel,height,width)
                # 2d images convert to 3d matrix
                frame = frame.transpose(2,0,1)
                frames.append(frame)
                f += 1
            else:
                break


        # Raw Video 길이를 6~10초로 선택.
        total_frames = f
        snippet_len = random.randint(6,10)

        snippet_start = random.randint(0, total_frames - 120) #random start 범위


        out = np.concatenate(frames)
        out = out.reshape(-1.3,270, 480)
        out = out[snippet_start : snippet_start+snippet_len*12 : , : , :]

        return self.transforms(out)

    def __len__(self):
        return len(self.videolist)



class Highlight_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot

        videolist = os.listdir(dataroot)
        self.videolist = [os.path.join(self.dataroot, video) for video in videolist  if os.path.splitext(video)[-1]=='.mp4']
        self.transforms = transform if transform is not None else lambda x: x


    def __getitem__(self, index):
        vidcap = cv2.VideoCapture(self.videolist[index])
        frames = []
        h_label = self.videolist[index].split("\\")[-2]


        while(vidcap.isOpened()):
            ret, frame = vidcap.read()

            if ret :
                frame = frame.transpose(2,0,1)
                frames.append(frame)
            else:
                break

        out = np.concatenate(frames) # 기존 array sequence 결합.
        out = out.reshape(-1, 3, 270, 480) # 전체를 rgb로 270*480으로 reshape.

        return self.transforms(out)


    def __len__(self):
        return len(self.videolist)

def video_transform(video, image_transform):
    # apply image transform to every frame in a video
    vid = []
    for img in video:
        vid.append(image_transform(img.transpose(1,2,0)))

    vid = torch.stack(vid) #stack에 쌓는다.
    # vid. 10, 3, 64, 64
    vid = vid.permute(1, 0, 2, 3)
    # vid. 3, 10, 64, 64

    return vid

def get_loader(h_dataroot, r_dataroot, batch_size):
    image_transforms = transforms.Compose([
    Image.fromarray,
    transforms.CenterCrop(270), # CentorCrop말고 그냥 reshape으로
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform = image_transforms)

    #각각의 dataset define
    h_dataset = Highlight_Dataset(h_dataroot, video_transforms)

    r_dataset = RawVideo_Dataset(r_dataroot, video_transforms)

    #Data Loader (input Pipeline)
    h_loader = DataLoader(h_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    r_loader = DataLoader(r_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    return h_loader, r_loader

if __name__=="__main__":
    get_loader("C:\\Users\JINI\workspace\DeepLearning\PROGRAPHY DATA\HV",
               "C:\\Users\JINI\workspace\DeepLearning\PROGRAPHY DATA\RV",1)