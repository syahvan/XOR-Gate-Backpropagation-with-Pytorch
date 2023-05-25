# Import library
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Buat data dummy untuk mensimulasikan input dan output XOR gate
x_data = torch.Tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]])

y_label = torch.Tensor([0., 1., 1., 0.]).reshape(x_data.shape[0], 1)

# Buat model neural network
class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.linear = nn.Linear(2, 2)
        self.Sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 1)

    def forward(self, input):
      x = self.linear(input)
      sig = self.Sigmoid(x)
      yh = self.linear2(sig)
      return yh

xor_network = XOR()
epochs = 5000
mseloss = nn.MSELoss()
optimizer = torch.optim.Adam(xor_network.parameters(), lr = 0.03)
all_losses = []
current_loss = 0

# Training data
for epoch in range(epochs):

  # input training example and return the prediction
  yhat = xor_network.forward(x_data)

  # calculate MSE loss
  loss = mseloss(yhat, y_label)
  
  # backpropogate through the loss gradiants
  loss.backward()

  # update model weights
  optimizer.step()

  # remove current gradients for next iteration
  optimizer.zero_grad()

  # append to loss
  current_loss += loss.item()
  all_losses.append(current_loss)
  current_loss = 0
  
  # print progress
  if epoch % 500 == 0:
    print(f'Epoch: {epoch} completed')

# Visualisasi animasi Neural Network Loss
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(0, epochs)
ax.set_ylim(0, max(all_losses))
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Neural Network Loss')

def update(frame):
    x = list(range(frame + 1))
    y = all_losses[:frame + 1]
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(all_losses), interval=1, blit=True)
plt.show()

# test input
input = torch.Tensor(x_data)
out = xor_network(input)
print("Hasil Prediksi:")
print(out.round())

