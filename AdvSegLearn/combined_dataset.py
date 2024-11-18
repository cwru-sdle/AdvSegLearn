import torch
from PIL import Image
class combined_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        transform=None,
        ):
        self.transform = transform
        self.data = data
        
    def __len__(self):
        return(self.data.__len__())
    def __getitem__(self,idx):
        print(self.data[idx])
        data_point = self.data[idx]
        if len(data_point) == 2:
            raw_image = Image.open(data_point[0]).convert("L")
            mask_image = Image.open(data_point[1]).convert("L")
            if self.transform:
                raw_image = self.transform(raw_image)
                mask_image = self.transform(mask_image)
            return raw_image, mask_image
        elif len(data_point)==1:
            raw_image = Image.open(data_point[0]).convert("L")
            if self.transform:
                raw_image = self.transform(raw_image)
            return raw_image
        elif isinstance(data_point, str):
            raw_image = Image.open(data_point).convert("L")
            if self.transform:
                raw_image = self.transform(raw_image)
            return raw_image
        else:
            raise Exception(f'Datapoint {idx} ie: {str(data_point)} is not a tuple with 1 or 2 elements.')