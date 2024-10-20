"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import numpy as np


# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
# create Network and Linear

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1(x).relu()
        end = self.layer2(middle).relu()
        return self.layer3(end).sigmoid()

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        # Retrieve weights and input rows
        weights = self.weights.value
       
        
        # Perform element-wise multiplication
        #weighted_sum = x_reshaped * weights_reshaped
        
        

        batch_size = x.shape[0]
        
        # Step 1: Reshape x to (batch_size, in_size, 1)
        x_reshaped = x.view([batch_size, self.in_size, 1])
        
        # Step 2: Reshape weights to (1, in_size, out_size)
        weights_reshaped = self.weights.value.view([1, self.in_size, self.out_size])
        

        #print("in_size: " +  str(self.in_size))
        #print("out_size: " +  str(self.out_size))
        #print(weights_reshaped.shape)
        #print(x_reshaped.shape)
        

        # Step 3: Broadcast and multiply
        intermediate = x_reshaped * weights_reshaped
        
        # Step 4: Sum along the input dimension (axis 1)
        output = intermediate.sum(1)
        
        # Step 5: Add bias
        output = output + self.bias.value

        # Collapse the output to (1, n)
        output = output.view([batch_size, self.out_size])

        #print(output)
        #print('------------')



        return output
    
"""
def forward(self, x):
        # Reshape x to (batch_size, in_size, 1) for proper broadcasting
        x_reshaped = x.view([x.shape[0], x.shape[1], 1])
        
        # Compute x * self.weights.value
        weighted_sum = x_reshaped * self.weights.value
        
        # Sum along dimension 2 (which is the original input dimension) and add bias
        output = weighted_sum.sum(1) + self.bias.value

        

        return output"""

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
