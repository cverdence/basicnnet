import torch
from sklearn.preprocessing import MinMaxScaler

input_train = torch.tensor(([0, 1, 0], [0, 1, 1], [0, 0, 0], [10, 0, 0], [10, 1, 1], [10, 0, 1]), dtype=torch.float)
output_train = torch.tensor(([0], [0], [0], [1], [1], [1]), dtype=torch.float)
input_pred = torch.tensor([1, 1, 0], dtype=torch.float)

input_test = torch.tensor(([1, 1, 1], [10, 0, 1], [0, 1, 10], [10, 1, 10], [0, 0, 0], [0, 1, 1]), dtype=torch.float)
output_test = torch.tensor(([0], [1], [0], [1], [0], [0]), dtype=torch.float)

scaler = MinMaxScaler()
input_train_scaled = torch.tensor(scaler.fit_transform(input_train), dtype=torch.float)
output_train_scaled = torch.tensor(scaler.fit_transform(output_train), dtype=torch.float)
input_test_scaled = torch.tensor(scaler.fit_transform(input_test), dtype=torch.float)
output_test_scaled = torch.tensor(scaler.fit_transform(output_test), dtype=torch.float)

torch.save(input_train_scaled, 'data/input_train_scaled.pt')
torch.save(output_train_scaled, 'data/output_train_scaled.pt')
torch.save(input_test_scaled, 'data/input_test_scaled.pt')
torch.save(output_test_scaled, 'data/output_test_scaled.pt')
torch.save(input_pred, 'data/input_pred.pt')