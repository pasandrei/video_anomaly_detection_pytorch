from PIL import Image
import torch.utils.data as data
import os
import torchvision.transforms as transforms
import torch


class UCSDAnomalyDataset(data.Dataset):
    """
    Dataset class to load  UCSD Anomaly Detection dataset
    Input:
    - root_dir -- directory shoud contain a folder for each video
    - time_stride (default 1) -- max possible time stride used for data augmentation
    - seq_len -- length of the frame sequence
    Output:
    - tensor of 10 normlized grayscale frames stiched together

    Note:
    [mean, std] for grayscale pixels is [0.3750352255196134, 0.20129592430286292]
    """

    def __init__(self, root_dir, seq_len=10, time_stride=1, transform=None):
        super(UCSDAnomalyDataset, self).__init__()
        self.root_dir = root_dir
        self.images_lists = []

        video_folders = self.__get_video_folders()

        # video_files = [file for file in os.listdir(self.root) if file.endswith((".mp4", ".avi"))]
        self.samples = []
        for folder_index, video_folder in enumerate(video_folders):
            path_to_folder = os.path.join(self.root_dir, video_folder)
            self.__update_images_lists(path_to_folder)

            frames_in_video = len(self.images_lists[-1])

            for t in range(1, time_stride + 1):
                for i in range(1, frames_in_video):
                    if i + (seq_len - 1) * t > frames_in_video:
                        break

                    self.samples.append((path_to_folder, range(i, i + (seq_len - 1) * t + 1, t), folder_index))

        self.pil_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.Grayscale(),
            transforms.ToTensor()])

        self.tensor_transform = transforms.Compose([
            transforms.Normalize(mean=(0.3750352255196134,), std=(0.20129592430286292,))])

    def __getitem__(self, index):
        sample = []
        path_to_folder = self.samples[index][0]
        folder_index = self.samples[index][2]
        images_list = self.images_lists[folder_index]

        for fr in self.samples[index][1]:
            with open(os.path.join(path_to_folder, images_list[fr - 1]), 'rb') as fin:
                frame = Image.open(fin).convert('RGB')
                frame = self.pil_transform(frame) / 255.0
                frame = self.tensor_transform(frame)
                sample.append(frame)

        sample = torch.stack(sample, dim=0)

        return [sample, path_to_folder]

    def __len__(self):
        return len(self.samples)

    def __get_video_folders(self):
        """
        Returns:
        The list with all the folders in self.root_dir. The folders which end in "_gt" are not included in this list
        because we do not want to run the anomaly detection on the images which contain the mask of the anomaly
        """
        folder_list = []

        for d in os.listdir(self.root_dir):
            if (os.path.isdir(os.path.join(self.root_dir, d)) and not d.endswith("_gt")):
                folder_list.append(d)

        return folder_list

    def __update_images_lists(self, path_to_image_folder):
        images_list = []

        for file in os.listdir(path_to_image_folder):
            if file.endswith((".tif", ".png", ".jpg")):
                images_list.append(file)

        self.images_lists.append(images_list)
