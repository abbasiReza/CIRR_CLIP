import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class CustomImageCaptionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, transform=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        target_image = Image.open(os.path.join('../nlvr/total/', row['tar_images'])).convert('RGB')
        ref_image = Image.open(os.path.join('../nlvr/total/', row['ref_images'])).convert('RGB')

        if self.transform:
            target_image = self.transform(target_image)
            ref_image = self.transform(ref_image)

        tokenized_caption = self.tokenizer(row['delta_captions'])
        return {
            'target_image': target_image,
            'ref_image': ref_image,
            'tokenized_caption': tokenized_caption.squeeze()
        }
