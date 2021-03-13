from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.data = dataset.data
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = self.transform(image)
        return (image, label)
