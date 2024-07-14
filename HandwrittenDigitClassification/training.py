from utils.model import *
#from utils.datasetloading import *
from  torch.utils.data import dataloader 
import torch
import torch.nn as nn
import torch.optim as optim

model = NeuralNetwork().to(device)
#HyperparameterNs
learning_rate = 0.001
batch_size = 64
epochs = 4
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=learning_rate)

#training loop
def train(dataloader,model,loss_fn,optimzer):
    size=len(dataloader.dataset)

    model.train()

    for batch,(X,y) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)
        #forward pass
        y_pred=model(X)
        #loss
        loss=loss_fn(y_pred,y)
        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

     
        if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#testing loop
def test(dataloader,model,lossfn):
     best_loss=float("inf")
     model.eval()
     size=len(dataloader.dataset)
     num_of_batches=len(dataloader)
     test_loss,correct_pred=0,0

 
     with torch.no_grad():
        for X,y in dataloader:
            X=X.to(device)
            y=y.to(device)
            pred=model(X)
            
            test_loss+=loss_fn(pred,y).item()
            correct_pred += (pred.argmax(1) == y).type(torch.float).sum().item()
            if test_loss<best_loss:
                best_loss=test_loss
                torch.save(model.state_dict(),"HandwrittenDigitClassification/model_weight.pth")
                
        test_loss /= num_of_batches
        correct_pred /= size
        print(f"Test Error: \n Accuracy: {(100*correct_pred):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        

for z in range(epochs):
    print(f"Epoch {z+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


model.load_state_dict(torch.load('HandwrittenDigitClassification/model_weight.pth'))
torch.save(model,'HandwrittenDigitClassification/model.pth')