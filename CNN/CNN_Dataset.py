from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from self_data_process import init_process
import numpy as np


class MyDataset(Dataset):
    """
    data:传入的数据集
    transform:处理图像数据
    loader:加载器
    """
    # 初始化需要的内容
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    # 最主要的是这个函数，它以迭代的方式加载数据集（image和label），再利用loader加载，最后用transform转换成Tensor类型。
    def __getitem__(self, item):
        """
        __getitem__是真正读取数据的地方，迭代器通过索引来读取数据集中数据，因此只需要这一个方法中加入读取数据的相关功能即可。
        在这个函数里面，我们对第二步处理得到的列表进行索引，
        接着利用第三步定义的Myloader来对每一个路径进行处理，
        最后利用pytorch的transforms对RGB数据进行处理，将其变成Tensor数据。
        :param item:
        :return:
        """
        image, label = self.data[item]
        image = self.loader(image)
        image = self.transform(image)
        return image, label

    # 返回数据的长度
    def __len__(self):
        return len(self.data)


def MyDatasetLoader(path):
    """
    利用PIL包的Image处理图片，把图像转换成RGB
    :param path:
    :return:
    """
    return Image.open(path).convert('RGB')


# 利用函数加载数据集
def load_cnn_data():
    print("data loading....")

    # 利用transformers对图像处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    training_cats = r'E:\Jann\Algorithm\Basic_Network_Architecture\data\CNN_Dataset\training_data\cats\cat.%d.jpg'
    training_cats_data = init_process(training_cats, [0, 500])

    # training_data里面的dog处理
    training_dogs = r'E:\Jann\Algorithm\Basic_Network_Architecture\data\CNN_Dataset\training_data\dogs\dog.%d.jpg'
    training_dogs_data = init_process(training_dogs, [0, 500])

    # testing_data里面的cat处理
    testing_cats = r'E:\Jann\Algorithm\Basic_Network_Architecture\data\CNN_Dataset\testing_data\cats\cat.%d.jpg'
    testing_cats_data = init_process(testing_cats, [1000, 1200])

    # testing_data里面的dog处理
    testing_dogs = r'E:\Jann\Algorithm\Basic_Network_Architecture\data\CNN_Dataset\testing_data\dogs\dog.%d.jpg'
    testing_dogs_data = init_process(testing_dogs, [1000, 1200])

    # 500+500+200+200=1400
    data = training_cats_data + training_dogs_data + testing_cats_data + testing_dogs_data

    # 打乱上述数据的顺序
    np.random.shuffle(data)

    # 划分数据集----我们只要训练和测试
    train_data, valid_data, test_data = data[:900], data[900:1100], data[1100:]

    # 数据集的加载和读取
    train_data = MyDataset(data=train_data, transform=transform, loader=MyDatasetLoader)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    valid_data = MyDataset(data=valid_data, transform=transform, loader=MyDatasetLoader)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True, num_workers=0)

    test_data = MyDataset(data=test_data, transform=transform, loader=MyDatasetLoader)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader
