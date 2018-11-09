
####################### SKeleton LSTM ######################

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


         def __init__(self, inp, out, lengths):

             self.x = inp
             self.y = out
             self.len = lengths

         def __getitem__(self,index ):

             return self.x[index], self.y[index], self.len[index]

         def __len__(self,):

              return self.x.size(0)

#pred_dir = '/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/lstm_predictions' ### path to folder where the predicted feats will be stored

X_train = []
Y_train = []

#ls =open('/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/slt_bdl/lstm_performance.txt','w')

source_folder ='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/input_full/'  ### input folder
os.chdir(source_folder)
files = sorted(os.listdir('.'))

for file in files:
        x_train = numpy.loadtxt(file)
        X_train.append(x_train)

target_folder ='/home3/srallaba/projects/siri_expts/8september/Data/vcc2018_training/SF1_TF1/output_full/'  ### output folder
os.chdir(target_folder)
files = sorted(os.listdir('.'))

for file in files:
#        print("file is:", file)
        y_train = numpy.loadtxt(file)
        Y_train.append(y_train)
         

X_train = np.array(X_train)
Y_train = np.array(Y_train)


#print("testing to be done on:", valid_files)
hidden = 128  ## assign the number of hidden units




class LSTMenc(nn.Module):

  def __init__(self, hidden_dim, hidden_layers, input_size, output_size):

    super(LSTMenc, self).__init__()

    self.hidden_dim = hidden_dim
    self.hidden_layers = hidden_layers
    self.input_size = input_size
    self.output_size = output_size

    self.lstm = nn.LSTM(input_size, self.hidden_dim, hidden_layers, batch_first=True)  #lstm = nn.LSTM(input_dim, lstmoutput/hidden_dim, num_of_layers)
    self.hidden2out = nn.Linear(self.hidden_dim, output_size)

  def init_hidden(self, batch):

    return  (autograd.Variable(torch.zeros(self.hidden_layers, batch, self.hidden_dim)),  # (num_layers * num_directions, batch size, hidden_size)
                autograd.Variable(torch.zeros(self.hidden_layers, batch, self.hidden_dim)))


  def forward(self, batch_in, lengths):
#      print("inputs len", batch_in.size(),lengths)
      self.hidden = self.init_hidden(batch_in.size(0))
      pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, lengths, batch_first=True)
      packed_output, (ht, ct) = self.lstm(pack, self.hidden)
      unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
      final_out = (self.hidden2out((unpacked)))
      return final_out


class LSTMdec(nn.Module):

  def __init__(self, hidden_dim, hidden_layers, input_size, output_size):

    super(LSTMdec, self).__init__()

    self.hidden_dim = hidden_dim
    self.hidden_layers = hidden_layers
    self.input_size = input_size
    self.output_size = output_size

    self.lstm = nn.LSTM(input_size, self.hidden_dim, hidden_layers, batch_first=True)  #lstm = nn.LSTM(input_dim, lstmoutput/hidden_dim, num_of_layers)
    self.hidden2out = nn.Linear(self.hidden_dim, output_size)

  def init_hidden(self, batch):

    return  (autograd.Variable(torch.zeros(self.hidden_layers, batch, self.hidden_dim)),  # (num_layers * num_directions, batch size, hidden_size)
                autograd.Variable(torch.zeros(self.hidden_layers, batch, self.hidden_dim)))


  def forward(self, bottleneck):
      self.hidden=self.init_hidden(bottleneck.size(0))
      output, (ht, ct) = self.lstm(bottleneck,self.hidden)
      final_out = (self.hidden2out((output)))
      return final_out


def pad_seq(sequence):

  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
 # print("seq len is:", seq_len)
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])), (0,0))
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths


def pad_seq_valid(sequence):

  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  lengths_v = [len(x) for x in sequence]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
  #print("seq len is:", seq_len)
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])), (0,0))
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths_v


def sort_batch(batch, ys, lengths):
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    targ_tensor = ys[perm_idx]
    return seq_tensor, targ_tensor, seq_lengths #

padded_input, lengths = pad_seq(X_train)
#print("lemgths r: aftr padding", lengths)
padded_output, lengths = pad_seq(Y_train)
data = arctic_data(torch.Tensor(padded_input), torch.Tensor(padded_output), lengths)

loss_fn = torch.nn.MSELoss(size_average=False)

#learning_rate = 1e-4

enc_model = LSTMenc(hidden, 3, 60, 10) ######hidden_dim, hidden_layers, input_size, output_size)
dec_model= LSTMdec(hidden, 3, 10, 60)
#optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
enc_optimizer = optim.Adagrad(enc_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
dec_optimizer = optim.Adagrad(dec_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= 0.9)

for i in range(0,30):
  train_loader = DataLoader(dataset = data, batch_size=4, shuffle=True)       
  total_loss = 0
  

  for in_batch, output, lengths in train_loader:
    new_lengths = []
#    print("input shape is:", numpy.shape(in_batch))
    in_batch, output, lengths = sort_batch(in_batch, output, lengths)
    for j in range(0,len(lengths)):
      new_lengths.append(lengths[j])
    bottleneck = enc_model(autograd.Variable(in_batch, requires_grad=True), new_lengths)
 #   print("intermediate feats are:", numpy.shape(bottleneck), numpy.shape(bottleneck[0]), numpy.shape(bottleneck[1]))


    pred = dec_model(bottleneck)
 #   print("predictions are:", numpy.shape(pred))

    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(output), new_lengths, batch_first=True)
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
 #   print("output data is of shape:", numpy.shape(unpacked))


    dec_loss = loss_fn(pred, unpacked)
    total_loss += dec_loss.data[0]
    dec_optimizer.zero_grad()
    dec_loss.backward()
    dec_optimizer.step()
  print("epoch:", i, "loss is:", total_loss/4) # divided by no. of examples, batch_size here

'''
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
