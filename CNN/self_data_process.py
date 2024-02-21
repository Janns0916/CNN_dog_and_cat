
from tqdm import tqdm


def init_process(path, lens):
    """
    得到一个包含所有图片文件名（包含路径）和标签（狗1猫0）的列表
    :param path: 猫狗数据集路径
    :param lens: 训练测试数据集数量
    :return: ['Basic_Network_Architecture/data/training_data/cats/cat.0.jpg', 0]
    """
    data = []
    name = find_label(path)
    for i in tqdm(range(lens[0], lens[1])):  # 0-500
        # 第一个为文件路径，第二个为路径中动物的label
        data.append([path % i, name])  # ['Basic_Network_Architecture/data/training_data/cats/cat.0.jpg', 0]

    return data


def find_label(str):  # str为图像路径名称

    """
    find_label来判断标签是dog还是cat
    :param str:
    :return: 0(cat)  or 1(dog)
    """

    # 这些是从路径中找到文件名中%d和cat、dog的索引
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        # 这个是找路径中d的索引
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        # 这个是找路径中是cat还是dog
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break
    # 找到之后，传入到name中，作为输出
    name = str[first:last]
    # 进行判断，然后找到之后，为其赋予label
    if name == 'dog':
        return 1
    else:
        return 0

# ***init_process和find_label针对不同的数据集进行不同的处理***


if __name__ == '__main__':
    """数据集的处理"""
    # # 现有数据的命名都是有序号的，训练集中数据编号为0-499，测试集中编号为1000-1200。
    # # 所以在处理数据集时，要针对训练集和测试集分别处理。里面还要单独处理cat和dog。
    #
    # # training_data里面的cat处理
    # training_cats = 'Basic_Network_Architecture/data/training_data/cats/cat.%d.jpg'
    # training_cats_data = init_process(training_cats, [0, 500])
    #
    # # training_data里面的dog处理
    # training_dogs = 'Basic_Network_Architecture/data/training_data/dogs/dog.%d.jpg'
    # training_dogs_data = init_process(training_dogs, [0, 500])
    #
    # # testing_data里面的cat处理
    # testing_cats = 'Basic_Network_Architecture/data/testing_data/cats/cat.%d.jpg'
    # testing_cats_data = init_process(testing_cats, [1000, 1200])
    #
    # # testing_data里面的dog处理
    # testing_dogs = 'Basic_Network_Architecture/data/testing_data/cats/dogs/dog.%d.jpg'
    # testing_dogs_data = init_process(testing_dogs, [1000, 1200])
    #
    # for folder_name in [training_cats_data, training_dogs_data, testing_cats_data,testing_dogs_data]:
    #     print("processing"+str(folder_name)+"...")
    #     print('\t')
    # print("Finishing...")
    pass