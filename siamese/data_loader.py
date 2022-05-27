import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset

#dataloader for getting triplets from dataset
class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        #img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0_tuple = self.imageFolderDataset.imgs[index]
        #search untille requirements are met (same class)
        while True:
          img1_tuple = random.choice(self.imageFolderDataset.imgs)
          if img0_tuple[1] == img1_tuple[1]:
            break
        while True:
          #search untill they arent the same class
          img2_tuple = random.choice(self.imageFolderDataset.imgs)
          if img0_tuple[1] != img2_tuple[1]:
            break
        while True:
          img3_tuple = random.choice(self.imageFolderDataset.imgs)
          if (img3_tuple[1] != img1_tuple[1]) and (img3_tuple[1] != img2_tuple[1]):
            break
        
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])
        img3 = Image.open(img3_tuple[0])

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)


        return img0, img1, img2, img3
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)