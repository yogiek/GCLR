import os, sys
from libs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        data_dir, 
        augment = False, 
    ):
        self.image_files = glob.glob(data_dir + "*/*")
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p = 0.5), 
                    A.Resize(
                        height = 128, width = 128, 
                    ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        height = 128, width = 128, 
                    ), 
                    A.Normalize(), AT.ToTensorV2(), 
                ]
            )

    def __len__(self, 
    ):
        return len(self.image_files)

    def __getitem__(self, 
        index, 
    ):
        image_file = self.image_files[index]
        image = cv2.imread(image_file)

        if image is None:
          raise ValueError(f"Image not found or is empty at index {index}: {image_file}")
        
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2RGB, 
        )
        # image, label, domain = self.transform(image = image)["image"], int(image_file.split("/")[-2]), int(image_file.split("/")[-3])
        
        image, label, domain = self.transform(image = image)["image"], image_file.split("/")[-2], image_file.split("/")[-3]

        return image, int(label), int(domain)