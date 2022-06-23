import torch
from torch import nn, tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader


class CNN(nn.Module):
    epoch = 1
    batch_size = 50
    lr = 0.001

    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )  # Define the first Convolutional Nerual Network
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten
        output = self.out(x)
        return output, x

def training(self, training_datas, test_datas):
    training_loader = DataLoader(
        dataset=training_datas, batch_size=self.batch_size, shuffle=True
    )
    test_x = (
        Variable(torch.unsqueeze(test_datas.test_data, dim=1), volatile=True).type(
            torch.FloatTensor
        )[:2000]
        / 255.0
    )  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_datas.test_labels[:2000]
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    loss_func = nn.CrossEntropyLoss()
    now = 0
    while now < self.epoch:
        # Training
        for idx, (x, y) in enumerate(training_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = self(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                # Test and virtualize the result in each 50 step
                test_output, last_layer = self(test_x)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy = sum(pred_y == test_y) / float(test_y.size(0))
                print(
                    "Epoch: ",
                    now,
                    "| train loss: %.4f" % loss.item(),
                    "| test accuracy: %.2f" % accuracy,
                )
                # if HAS_SK:
                #     # Visualization of trained flatten layer (T-SNE)
                #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                #     plot_only = 500
                #     low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                #     labels = test_y.numpy()[:plot_only]
                #     plot_with_labels(low_dim_embs, labels)
        now += 1
