import matplotlib.pyplot as plt
import torchvision


class MnistHandWroteNumber:
    @classmethod
    def download(cls, path: str = "", train: bool = True, force: bool = False) -> tuple:
        train_data = torchvision.datasets.MNIST(
            root=path,
            train=train,
            transform=torchvision.transforms.ToTensor(),
            download=(not force),
        )
        plt.imshow(train_data.train_data[0].numpy(), cmap="gray")
        plt.title(train_data.train_labels[0])
        plt.show()
        return train_data, train_data.train_data.size()
