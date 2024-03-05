from torchvision.datasets import CIFAR10
from PIL import Image

class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, focused_transform=None, focused_classes=None):
        super().__init__(root, train=train, download=download, transform=transform)
        self.focused_transform = focused_transform
        # self.focused_classes = focused_classes or []

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        image = Image.fromarray(image)
        
        # Apply focused transform to specified classes
        # if label in self.focused_classes and self.focused_transform is not None:
        if self.focused_transform is not None:
            image = self.focused_transform(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
