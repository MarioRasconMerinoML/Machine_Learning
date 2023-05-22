from torch.utils.data import Dataset, DataLoader

class TextDatasetFromDir(Dataset):
    '''
    This class is equivalent to keras.utils.text_dataset_from_directory
    
    it helps to load every dataset from the pre-configured folder structure like:
    main_directory/
        ...class_a/
        ......a_text_1.txt
        ......a_text_2.txt
        ...class_b/
        ......b_text_1.txt
        ......b_text_2.txt
    
    '''
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.classes = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            with tqdm(total=12000) as pbar:
                for filename in os.listdir(class_dir):
                    path = os.path.join(class_dir, filename)
                    if not os.path.isfile(path):
                        continue
                    if self._has_valid_extension(filename):
                        item = (path, self.class_to_idx[class_name])
                        samples.append(item)
                        pbar.update()
        return samples

    def _has_valid_extension(self, filename):
        valid_extensions = ['.txt']  # Add more extensions if needed
        return any(filename.endswith(ext) for ext in valid_extensions)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
