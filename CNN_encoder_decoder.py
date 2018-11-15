
####################### SKeleton CNN ######################

import os
import numpy
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class arctic_data(Dataset):


         def __init__(self, inp, out):

             self.x = inp
             self.y = out
             #self.len = lengths

         def __getitem__(self,index ):

             return self.x[index], self.y[index]#, self.len[index]

         def __len__(self,):

              return self.x.size(0)

pred_dir = '/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/CNN_predictions' ### path to folder where the predicted feats will be stored

X_train = []
Y_train = []

#ls =open('/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/lstm_performance.txt','w')

source_folder ='/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/input_full/'  ### input folder
os.chdir(source_folder)
files = sorted(os.listdir('.'))

for file in files:
        x_train = numpy.loadtxt(file)
        X_train.append(x_train)

target_folder ='/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/output_full/'  ### output folder
os.chdir(target_folder)
files = sorted(os.listdir('.'))

for file in files:
#        print("file is:", file)
        y_train = numpy.loadtxt(file)
        Y_train.append(y_train)
         

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("see meeeeeeee", numpy.shape(X_train))


class my_cnn_encoder(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
   super(my_cnn_encoder,self).__init__()
   self.in_channels = in_channels
   self.out_channels = out_channels
   self.kernel_size = kernel_size
   self.stride = stride
   self.padding = padding

   self.cnn=nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
   self.batchnorm = nn.BatchNorm2d(self.out_channels)
   self.relu = nn.ReLU()

  def forward(self,x):
   out = self.cnn(x)
   out = self.bacthnorm(out)
   out = self.relu(out)
   return out


class my_cnn_decoder(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
   super(my_cnn_decoder,self).__init__()
   self.in_channels = in_channels
   self.out_channels = out_channels
   self.kernel_size = kernel_size
   self.stride = stride
   self.padding = padding

   self.cnn=nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
   self.batchnorm = nn.BatchNorm2d(self.out_channels)
   self.relu = nn.ReLU()

  def forward(self,x):
   out = self.cnn(x)
   out = self.bacthnorm(out)
   out = self.relu(out)
   return out

class EncDec(nn.Module):

    def __init__(self, encoder, decoder):
          super(EncDec,self).__init__()
          self.encoder = encoder
          self.decoder = decoder
    def forward(self,batch_in, lengths):
         bottleneck = self.encoder(   batch_in, lengths )
         return self.decoder(bottleneck)

#X_train= X_train.astype(dtype= 'float32')
#Y_train=Y_train.astype(dtype='float32')  # https://discuss.pytorch.org/t/cant-convert-a-given-np-ndarray-to-a-tensor/21321/2
#data = arctic_data(Variable(torch.from_numpy(X_train)), Variable(torch.from_numpy(Y_train)))

loss_fn = torch.nn.MSELoss(size_average=False)

#learning_rate = 1e-4

enc_model = my_cnn_encoder(180, 50, 5,1,2) ######hidden_dim, hidden_layers, input_size, output_size)
dec_model= my_cnn_decoder(50,180, 5, 1,2)
encdec = EncDec(enc_model, dec_model)
optimizer = optim.Adagrad(encdec.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
#enc_optimizer = optim.Adagrad(enc_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
#dec_optimizer = optim.Adagrad(dec_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= 0.9)

for i in range(0,30):
  #train_loader = DataLoader(dataset = data, batch_size=4, shuffle=True)       
  total_loss = 0
  

  for (src, tgt) in zip(X_train, Y_train):
   
#    print("input shape is:", numpy.shape(in_batch))
   # in_batch, output, lengths = sort_batch(in_batch, output, lengths)
    #for j in range(0,len(lengths)):
    #  new_lengths.append(lengths[j])
    pred = encdec(autograd.Variable(torch.tensor(src), torch.tensor(tgt)))
    print("predictions:", pred)
 #   print("intermediate feats are:", numpy.shape(bottleneck), numpy.shape(bottleneck[0]), numpy.shape(bottleneck[1]))
'''

    #pred = dec_model(bottleneck)
 #   print("predictions are:", numpy.shape(pred))

    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(output), new_lengths, batch_first=True)
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
 #   print("output data is of shape:", numpy.shape(unpacked))


    loss = loss_fn(pred, unpacked)
    total_loss += loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print("epoch:", i, "loss is:", total_loss/4) # divided by no. of examples, batch_size here


  msg = str(i) +' ' + str(total_loss/4)
#  print("msg is:", msg)
  ls.write(str(msg)+ '\n')
ls.close()
torch.save(model, '/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/lstm_model.pt')  #### path to save the model
#print("vaid on:", X_test)
padded_valid_in, lengths = pad_seq_valid(X_valid)
padded_valid_out, lengths = pad_seq_valid(Y_valid)
data = arctic_data(torch.Tensor(padded_valid_in), torch.Tensor(padded_valid_out),lengths)
valid_loader = DataLoader(dataset = data, batch_size=1)
ii =  0
for in_batch, output, lengths in valid_loader:
    new_lengths = []
##    in_batch, output, lengths = sort_batch(in_batch, output, lengths)
#    print("len is:", ii)
#    print("in validation in batch is..............", in_batch.size())
    for i in range(0,len(lengths)):
      new_lengths.append(lengths[i])
    Model = torch.load('/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/lstm_model.pt')   ### path to load the model from
    print("loading model")
    pred = Model(autograd.Variable(in_batch, requires_grad=False), new_lengths)
    
#    print("pred is", pred.size())
    print("writing the predicted validation file for....", valid_files[ii])
    np.savetxt(pred_dir + '/'+ valid_files[ii], pred.data[0])
    ii = ii+1

'''
