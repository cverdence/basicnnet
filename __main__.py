from basicnnet.neural_network import NeuralNetwork
import torch

input_train_scaled = torch.load('data/input_train_scaled.pt')
output_train_scaled = torch.load('data/output_train_scaled.pt')
input_test_scaled = torch.load('data/input_test_scaled.pt')
output_test_scaled = torch.load('data/output_test_scaled.pt')
input_pred = torch.load('data/input_pred.pt')

NN = NeuralNetwork()
NN.train(input_train_scaled, output_train_scaled, 200)
NN.predict(input_pred)
NN.view_error_development()
NN.test_evaluation(input_test_scaled, output_test_scaled)