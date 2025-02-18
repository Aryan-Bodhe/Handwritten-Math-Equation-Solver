import sys
import os

# Ensure the parent directories are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import torch

from Preprocessing.Preprocessing import preprocess_image_from_input
from model.architecture import CNN

class CustomDataset():
    def __init__(self, root_dir, transform=None):
        print("\n\nInitializing Custom Dataset")
        self.root_dir = root_dir
        print("Root Directory:", root_dir)
        # # Get all subfolders (class labels)
        # self.classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]

        # self.class_to_idx = {cls_name: inverted_class_mapping[cls_name] for cls_name in self.classes if cls_name in inverted_class_mapping}

        # Initialize list of image paths and corresponding labels
        self.image_paths = []
        

        # Loop through all subfolders and images
        
        class_folder = os.path.join(root_dir)
        print("Class Folder:", class_folder)
        if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    print("Image Path:", img_path)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):  # Ensure valid image formats
                        self.image_paths.append(img_path)
                        
                print("Image Processed")

    def __len__(self):
        return len(self.image_paths)

    def getitem(self):
        """ Get an image and its corresponding label """
        images = []
        self.image_paths = sorted(self.image_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for i in self.image_paths:
            print("Processing Image Path:", i)
            image = cv2.imread(i, cv2.IMREAD_GRAYSCALE) # Do not convert to RGB
            image = torch.tensor(image).unsqueeze(0)
            images.append(image)
            
        print("\n\nImages: ", len(images))

        return images


inverted_class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "+",
    37: "-",
    38: "forward_slash",
}


def runcode():
    # Get the image path from the user
    print('Enter image path, (CTRL+C to exit):', end='')
    image_path = str(input())

    # Check if the path is valid
    if not os.path.exists(image_path):
        print('Invalid path.')
        exit()

    # Preprocess the image
    preprocess_image_from_input(image_path)


        
    test_dataset = CustomDataset(root_dir=os.path.join(os.getcwd(), "cache/final_letters"))
    test_dataset_x = test_dataset.getitem()

    test_dataset_x = torch.stack(test_dataset_x).cuda()

    # Load the trained model
    model = CNN().cuda()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    test_dataset_x = test_dataset_x.to(torch.float32).cuda()

    output = model(test_dataset_x)
    pred = torch.argmax(output, dim=1)

    pred = [inverted_class_mapping[p.item()] for p in pred]

    print("Predicted Equation: ", end='')
    for p in pred:
        print(p, end='')
    print()


    checkpoint = torch.load("model.pth")
    print(checkpoint.keys())
    #help(checkpoint)

while(True):
    runcode()

# #Delete final_letters directory
# import shutil
# shutil.rmtree(os.path.join(os.getcwd(), "cache/final_letters"))
# shutil.rmtree(os.path.join(os.getcwd(), "cache/image.png"))
# shutil.rmtree(os.path.join(os.getcwd(), "cache/deskewed_image.png"))
# shutil.rmtree(os.path.join(os.getcwd(), "cache/cropped_image.png"))