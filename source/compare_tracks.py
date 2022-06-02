import random
import os
from PIL import Image

#nog lelijke rommelcode

class Pairwise(Dataset):
    def __init__(self, img1, img2, transform):
        self.img1 = img1
        self.img2 = img2
        self.transform = transform

    def __getitem__(self, index):
        img1 = self.transform(self.img1)
        img2 = self.transform(self.img2)
        return img1, img2

    def __len__(self):
        return 2

#amount 0 if you want to compare only 2  random images of every set
def testimages(track_1, track_2, amount=0, transform=transformation):
  """
  track_1/2: path of two tracks that are compared
  index: index of image that is compared of the track
  amount: amount of images that are compared
  Shuffle: if you want the images that it selects to be shuffled
  """

  #images in the track folder
  trackfolder1 = os.listdir(track_1)
  trackfolder2 = os.listdir(track_2)

  #compare only two random images
  if amount == 0:
    #selects images based on the index
    image1 = random.choice(trackfolder1)
    image2 = random.choice(trackfolder2)
    image1 = Image.open(track_1+'/'+image1)
    image2 = Image.open(track_2+'/'+image2)
    pairwise = Pairwise(image1, image2,transform=transformation)
    pair = DataLoader(
            pairwise,
            num_workers=1,
            batch_size=1,
            shuffle=False
            )
    dataiter = iter(pair)
    image11, image22= next(dataiter)

    #send images to the model to be compared
    (vector1, vector2,_,_) =  net(image11.cuda(), image22.cuda())

    #output the distance
    distance = F.pairwise_distance(vector1, vector2)
    #plot 2 images next to eachother 
    # f = plt.figure()
    # f.add_subplot(1,2, 1)
    # plt.imshow(image1)
    # f.add_subplot(1,2, 2)
    # plt.imshow(image2)
    # plt.show(block=True)
  
    print(distance.item())
    if distance.item() <= 1:
      print('same person')
      print(f'Tracks,{track_1} and {track_2} contain the same person')
    else:
      print('different person')
      print(f'Tracks,{track_1} and {track_2} don\'t contain the same person')

  
  else: 
    dist = 0
    #compares multiple images
    for i in range(amount):
      image1 = random.choice(trackfolder1)
      image2 = random.choice(trackfolder2)
      image1 = Image.open(track_1+'/'+image1)
      image2 = Image.open(track_2+'/'+image2)
      pairwise = Pairwise(image1, image2,transform=transformation)
      pair = DataLoader(
            pairwise,
            num_workers=1,
            batch_size=1,
            shuffle=False
            )
      dataiter = iter(pair)
      image11, image22= next(dataiter)

      (vector1, vector2,_,_) =  net(image11.cuda(), image22.cuda())

      distance = F.pairwise_distance(vector1, vector2)

      #keeps the distances to create an average over the total amount of images that are compared
      dist += distance.item()
      average_distance = dist/amount
      
      # f = plt.figure()
      # f.add_subplot(1,2, 1)
      # plt.imshow(image1)
      # f.add_subplot(1,2, 2)
      # plt.imshow(image2)
      # plt.show(block=True)
  
      print(f'Distance set {i} = ', distance.item())
    print("Average distance = ", average_distance)
    if average_distance <= 1:
      print(f'Tracks,{track_1} and {track_2} contain the same person \n')
    else:
      print(f'Tracks,{track_1} and {track_2} don\'t contain the same person \n')

#compare all tracks in directory with the one you select
for i in os.listdir('/content/drive/MyDrive/faces/test/')[:-1]:
  testimages('/content/drive/MyDrive/faces/test/1','/content/drive/MyDrive/faces/test/'+i,5)


####for specific images
# def testimages(track_1, track_2, index, amount=0, transform=transformation, shuffle=True):
#   """
#   track_1/2: path of two tracks that are compared
#   index: index of image that is compared of the track
#   amount: amount of images that are compared
#   Shuffle: if you want the images that it selects to be shuffled
#   """

#   #images in the track folder
#   trackfolder1 = os.listdir(track_1)
#   trackfolder2 = os.listdir(track_2)

#   #if amount = 0 
#   if amount == 0:
#     #selects images based on the index
#     image1 = trackfolder1[index]
#     image2 = trackfolder2[index]
#     image1 = Image.open(track_1+'/'+image1)
#     image2 = Image.open(track_2+'/'+image2)
#     pairwise = Pairwise(image1, image2,transform=transformation)
#     pair = DataLoader(
#             pairwise,
#             num_workers=1,
#             batch_size=1,
#             shuffle=False
#             )
#     dataiter = iter(pair)
#     image11, image22= next(dataiter)

#     #send images to the model to be compared
#     (vector1, vector2,_,_) =  net(image11.cuda(), image22.cuda())

#     #output the distance
#     distance = F.pairwise_distance(vector1, vector2)
#     #plot 2 images next to eachother 
#     f = plt.figure()
#     f.add_subplot(1,2, 1)
#     plt.imshow(image1)
#     f.add_subplot(1,2, 2)
#     plt.imshow(image2)
#     plt.show(block=True)
  
#     print(distance.item())
#     if distance.item() <= 1:
#       print('same person')
#       print(f'Tracks,{track_1} and {track_2} contain the same person')
#     else:
#       print('different person')
#       print(f'Tracks,{track_1} and {track_2} don\'t contain the same person')