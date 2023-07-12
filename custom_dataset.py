import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class NumDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        # if train:
        #     self.cat_path = path + '/0'
        #     self.dog_path = path + '/dog/train'
        # else:
        #     self.cat_path = path + '/cat/test'
        #     self.dog_path = path + '/dog/test'
        
        self.zero_path = path + '/0'
        self.zero_img_list = glob.glob(self.zero_path + '/*.png')
        self.one_path = path + '/1'
        self.one_img_list = glob.glob(self.one_path + '/*.png')
        self.two_path = path + '/2'
        self.two_img_list = glob.glob(self.two_path + '/*.png')
        self.three_path = path + '/3'
        self.three_img_list = glob.glob(self.three_path + '/*.png')
        self.four_path = path + '/4'
        self.four_img_list = glob.glob(self.four_path + '/*.png')
        self.five_path = path + '/5'
        self.five_img_list = glob.glob(self.five_path + '/*.png')
        self.six_path = path + '/6'
        self.six_img_list = glob.glob(self.six_path + '/*.png')
        self.seven_path = path + '/7'
        self.seven_img_list = glob.glob(self.seven_path + '/*.png')
        self.eight_path = path + '/8'
        self.eight_img_list = glob.glob(self.eight_path + '/*.png')
        self.nine_path = path + '/9'
        self.nine_img_list = glob.glob(self.nine_path + '/*.png')


        self.transform = transform

        self.img_list = (self.zero_img_list + self.one_img_list + self.two_img_list + self.three_img_list 
                        + self.four_img_list + self.five_img_list + self.six_img_list 
                        + self.seven_img_list + self.eight_img_list + self.nine_img_list)

        self.class_list = ([0] * len(self.zero_img_list)+ [1] * len(self.one_img_list) + [2]*len(self.two_img_list )
                           + [3] * len(self.three_img_list) + [4] * len(self.four_img_list) + [5]*len(self.five_img_list)
                           + [6] * len(self.six_img_list) + [7] * len(self.seven_img_list) + [8]*len(self.eight_img_list)
                           + [9] * len(self.nine_img_list)
                           )
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = NumDataset(path='./custom_data', train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset,
                        batch_size=3,
                        shuffle=False,
                        drop_last=False)

    for epoch in range(10):
        print(f"epoch : {epoch} ")
        for batch in dataloader:
            img, label = batch
            print(img.size(), label)