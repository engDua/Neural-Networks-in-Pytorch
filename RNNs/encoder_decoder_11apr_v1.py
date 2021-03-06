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

class blizzard_data(Dataset):


         def __init__(self, inp, out, lengths):

             self.x = inp
             self.y = out
             self.len = lengths

         def __getitem__(self,index ):

             return self.x[index], self.y[index], self.len[index]

         def __len__(self,):

              return self.x.size(0)

test_dir = '/home3/srallaba/projects/siri_expts/21march_learning/test'

X_train = []
Y_train = []
X_test = []
Y_test = []
valid_files = []

g = open('files.test','w')
source_folder='/home3/srallaba/projects/siri_expts/scripts/input_full/'
k = 0
os.chdir(source_folder)
files = sorted(os.listdir('.'))
for file in files:
  if k < 8:
     k = k+1

     if '9' in file:
      x_valid = numpy.loadtxt(file)
      X_test.append(x_valid)

      g.write(file.split('.')[0] + '\n')

     else:
      x_train = numpy.loadtxt(file)
      X_train.append(x_train)
print("input shape is:", numpy.shape(X_train))


target_folder ='/home3/srallaba/projects/siri_expts/scripts/output_full/'


os.chdir(target_folder)
gfiles = sorted(os.listdir('.'))
k = 0
for gfile in gfiles:
 if k< 8:
   k = k+1
   if '9' in gfile:
    y_valid = numpy.loadtxt(gfile)
    Y_test.append(y_valid)
    valid_files.append(gfile)

   else:
     y_train = numpy.loadtxt(gfile)
     Y_train.append(y_train)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test = np.array(X_test)
Y_test = np.array(Y_test)

#print("testing to be done on:", valid_files)
hidden_dim = 64




class LSTMpred(nn.Module):

  def __init__(self, hidden_dim, hidden_layers, input_size, output_size):

    super(LSTMpred, self).__init__()

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
#      print("unpacked len",unpacked.size(-2),  unpacked_len)
#      self.bn= nn.BatchNorm2d(unpacked.size(-2))    
      final_out = (self.hidden2out((unpacked)))
#      print(final_out)
      return final_out

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, input_size):

        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def init_hidden(self,layers, batch_size):

     return  autograd.Variable(torch.zeros(layers, batch_size, self.hidden_size))


    def forward(self, batch_in, lengths, hidden):
#      self.hidden = self.init_hidden(batch_in.size(0))
      pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, lengths, batch_first=True)
      packed_output, ht = self.gru(pack, hidden)
      unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
      unpacked_outputs = unpacked[:, :, :self.hidden_size]
#      print("before", numpy.shape(unpacked))
#      print("later", numpy.shape(unpacked_outputs))
      return unpacked_outputs, ht

class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
#        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch):

      return  autograd.Variable(torch.zeros(self.hidden_layers, batch, self.hidden_dim))
    
    def forward(self, input,lengths, hidden):
#        output = self.embedding(input).view(1, 1, -1)
        pack = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
        input = F.relu(unpacked)
        output, hidden = self.gru(input, hidden)
        final_out = self.out(output)
        return final_out, hidden




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
# utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, if you use batch_first=True
# Otherwise, give (L,B,D) tensors --> https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

padded_input, lengths = pad_seq(X_train)
#print("lemgths r: aftr padding", lengths)
padded_output, lengths = pad_seq(Y_train)
data = blizzard_data(torch.Tensor(padded_input), torch.Tensor(padded_output), lengths)
train_loader = DataLoader(dataset = data, batch_size=4, shuffle=True)

loss_fn = torch.nn.MSELoss(size_average=False)

#learning_rate = 1e-4

enc_model = EncoderRNN(hidden_dim,  711) # hidden_dim, input_dim
dec_model = DecoderRNN(hidden_dim, 60)
#model = LSTMpred(hidden, 3, 711, 60) #hidden_dim, hidden_layers, input_size, output_size)
optimizer = optim.Adam(enc_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

encoder_hidden = enc_model.init_hidden(1,4) # num of layers, batch_size
#print("aftr initilatisatoon", numpy.shape(encoder_hidden))
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= 0.9)

for i in range(0,30):
  total_loss = 0
  print("epoch", i)

  for in_batch, output, lengths in train_loader:
    new_lengths = []
    in_batch, output, lengths = sort_batch(in_batch, output, lengths)
    for i in range(0,len(lengths)):
      new_lengths.append(lengths[i])
   
    enc_output, encoder_hidden = enc_model(autograd.Variable(in_batch, requires_grad=True), new_lengths, encoder_hidden)
    print("encoder output is", numpy.shape(enc_output))

    decoder_hidden = encoder_hidden
    dec_output, decoder_hidden = dec_model(autograd.Variable(in_batch, requires_grad=True), new_lengths, decoder_hidden)
    print("dec output is", numpy.shape(dec_out))
# write self. init hidden here
# pack the decoder input den unpack it and den perfrom decodong  --> this is because aftr unpacking encoder outputs have the max length in the batch as the seq length
# initilaise hidden for decoder

'''
    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(output), new_lengths, batch_first=True)
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
    loss = loss_fn(pred, unpacked)
    total_loss += loss.data[0]
# ============ make zero grad for both enc n decoder ========    ####optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print("loss is:", total_loss)

torch.save(model, '/home/siri/Documents/Projects/scripts/PyTorch/learning/21march_learning/model.pt')
## dont get confused by batch size and batch
print("vaid on:", X_test)
padded_valid_in, lengths = pad_seq_valid(X_test)
padded_valid_out, lengths = pad_seq_valid(Y_test)
data = blizzard_data(torch.Tensor(padded_valid_in), torch.Tensor(padded_valid_out),lengths)
valid_loader = DataLoader(dataset = data, batch_size=1)

ii =  0
for in_batch, output, lengths in valid_loader:
    new_lengths = []
##    in_batch, output, lengths = sort_batch(in_batch, output, lengths)
    print("len is:", ii)
    print("in validation in batch is..............", in_batch.size())
    for i in range(0,len(lengths)):
      new_lengths.append(lengths[i])
    Model = torch.load('/home/siri/Documents/Projects/scripts/PyTorch/learning/21march_learning/model.pt')
    print("loading model")
    pred = Model(autograd.Variable(in_batch, requires_grad=False), new_lengths)
    
    print("pred is", pred.size())
    print("writing", valid_files[ii])
    np.savetxt(test_dir + '/'+ valid_files[ii], pred.data[0])
    ii = ii+1

'''
