import torch
import torch.nn.functional as F
import torch.nn as nn


class MLPClassifier(nn.Module):
    # 单隐层感知机
    hidden_layer_size = 256

    def __init__(self, output_size, input_size=4096):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, MLPClassifier.hidden_layer_size),
            nn.BatchNorm1d(MLPClassifier.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(MLPClassifier.hidden_layer_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        classification = self.classifier(x)
        return classification


class PredictTime(nn.Module):
    hidden_layer_size = 256

    def __init__(self, output_size, input_size=4096):
        super(PredictTime, self).__init__()

        self.predict = nn.Sequential(
            nn.Linear(input_size, PredictTime.hidden_layer_size),
            nn.BatchNorm1d(PredictTime.hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(PredictTime.hidden_layer_size, output_size)
        )

    def forward(self, x):
        prediction = self.predict(x)
        return prediction


class ALEXNetMOD(torch.nn.Module):
    def __init__(self, time_out_size, fre_out_size, class_num):
        super(ALEXNetMOD, self).__init__()

        self.time_out_size = time_out_size
        self.fre_out_size = fre_out_size

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=7, padding=3),  #
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  #
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=5, padding=2),  #
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, padding=1),  #
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(384, 384, kernel_size=3, padding=1),  # [batch, 384, 12, 12]
            nn.BatchNorm1d(384),
            nn.ReLU(),

            nn.Conv1d(384, 256, kernel_size=3, dilation=1, padding=1),  # [batch, 256, 12, 12]
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),  # [batch, 256, 12, 12]

            nn.Conv1d(256, 4096, kernel_size=7, padding=3),  # 7 3
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Conv1d(4096, 4096, kernel_size=1, dilation=1),  # [batch, 4096, 12, 12]
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.flatten = torch.nn.Flatten()
        self.classification = MLPClassifier(input_size=4096, output_size=class_num)
        self.prediction_time = PredictTime(input_size=4096, output_size=time_out_size)  #
        self.prediction_fre = PredictTime(input_size=4096, output_size=fre_out_size)  #

    def forward(self, original_time_series, frequencies_of_subsequence, subsequence):
        c = self.layer1(original_time_series)
        c = self.layer2(c)
        c = self.layer3(c)
        c = self.layer4(c)
        c = self.flatten(c)
        c = F.normalize(c, dim=1)
        classification = self.classification(c)

        f = self.layer1(frequencies_of_subsequence)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        f = self.flatten(f)
        f = F.normalize(f, dim=1)
        fre_predict = self.prediction_fre(f)

        t = self.layer1(subsequence)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.layer4(t)
        t = self.flatten(t)
        t = F.normalize(t, dim=1)
        time_predict = self.prediction_time(t)

        return classification, fre_predict, time_predict

    def forward_test(self, original_time_series):
        c = self.layer1(original_time_series)
        c = self.layer2(c)
        c = self.layer3(c)
        c = self.layer4(c)
        c = self.flatten(c)
        c = F.normalize(c, dim=1)
        classification = self.classification(c)
        return classification


def optimize_network(model, optimizer, batch_original_time_series, batch_class_label,
                     batch_subsequence, batch_subsequence_label, batch_frequencies_of_subsequence,
                     batch_frequencies_of_sunsequence_label, alpha=0.1):
    batch_classification_hat, batch_frequencies_of_subsequence_hat, batch_subsequence_hat = model(
        batch_original_time_series.float(), batch_frequencies_of_subsequence.float(),
        batch_subsequence.float())

    batch_frequencies_of_sunsequence_label = batch_frequencies_of_sunsequence_label.squeeze(dim=1)
    batch_subsequence_label = batch_subsequence_label.squeeze(dim=1)

    loss_forecasting_fre = nn.MSELoss()(batch_frequencies_of_subsequence_hat,
                                        batch_frequencies_of_sunsequence_label.float())
    loss_forecasting_time = nn.MSELoss()(batch_subsequence_hat, batch_subsequence_label.float())
    loss_classification = nn.CrossEntropyLoss()(batch_classification_hat, batch_class_label.long())

    loss_forecast = loss_forecasting_fre + loss_forecasting_time
    loss_mtl = alpha * loss_forecast + (1 - 2 * alpha) * loss_classification
    optimizer.zero_grad()  # 优化器清零
    loss_mtl.backward()  # 求梯度
    optimizer.step()  # 更新模型参数
    return loss_classification.item(), loss_forecast.item()
