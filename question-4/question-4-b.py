import torch
import torch.nn as nn

def sigmoid_activation(x):

  return 1 / (1 + torch.exp(-x))

def tan_h_activation(x):
  response = (torch.exp(x)-torch.exp(-x)) / (torch.exp(x)+torch.exp(-x))
  return response

# Girdimiz input layer
our_matris = [
    [1,2,3],
    [4,5,6]
]

# Sonuç aynı çıksın diye ekliyoruz
torch.manual_seed(190401071)
# Tensöre çevirme işlemi
a = torch.Tensor(our_matris)
# hidden layer
hidden_layer = nn.Linear(3,50)
hidden_output = tan_h_activation(hidden_layer(a))
# output layer
output_layer = nn.Linear(50,1)
output = sigmoid_activation(output_layer(hidden_output))
print(output)


