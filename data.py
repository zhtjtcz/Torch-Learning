import torchvision as tv
from torch.utils.data import DataLoader

MNIST_img_size = 28
MNIST_img_ch = 1
MNIST_num_classes = 10
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)    # 逗号是为了让他是tuple；如果没有逗号就是括号了，得到的值是int类型的


def get_dataloaders(data_root_path: str, batch_size: int):
    transform_train = tv.transforms.Compose([
        tv.transforms.RandomCrop(size=MNIST_img_size, padding=4),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    transform_test = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    
    train_set = tv.datasets.MNIST(
        root=data_root_path,
        train=True, download=True,  # 注意train参数传的值
        transform=transform_train
    )
    test_set = tv.datasets.MNIST(
        root=data_root_path,
        train=False, download=True, # 注意train参数传的值
        transform=transform_test
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,               # 注意shuffle参数传的值
        num_workers=2,              # 多进程加载数据
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,              # 注意shuffle参数传的值
        num_workers=2,              # 多进程加载数据
    )
    
    return train_loader, test_loader
