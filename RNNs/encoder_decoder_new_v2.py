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

test_dir = '/home/siri/Documents/Projects/NUS_projects/VAE_stuff/test'

X_train = []
Y_train = []
X_test = []
Y_test = []
valid_files = []

g = open('files.test','w')
source_folder='/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/bdl_slt/input_full/'
k = 0
os.chdir(source_folder)
files = sorted(os.listdir('.'))
for file in files:
  if k < 71:
     k = k+1

     if '9' in file:
      x_valid = numpy.loadtxt(file)
      X_test.append(x_valid)

      g.write(file.split('.')[0] + '\n')

     else:
      x_train = numpy.loadtxt(file)
      X_train.append(x_train)
print("input shape is:", numpy.shape(X_train))


target_folder ='/home/siri/Documents/Projects/NUS_projects/vc_arctic_data/warped_feats/bdl_slt/input_full/'


os.chdir(target_folder)
gfiles = sorted(os.listdir('.'))
k = 0
for gfile in gfiles:
 if k< 71:
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


class EncoderGRU(nn.Module):
    def __init__(self, hidden_size, input_size):

        super(EncoderGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, 3) # input size, hidden size, num of layers

    def init_hidden(self,layers, batch_size):

     return  autograd.Variable(torch.zeros(layers, batch_size, self.hidden_size)) # num of layers* directions, batch, hidden_size


    def forward(self, batch_in, lengths, hidden):
      pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, lengths, batch_first=True)
      packed_output, ht = self.gru(pack, hidden)
      unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
      return unpacked, ht

class DecoderGRU(nn.Module):

    def __init__(self, hidden_size, output_size):

        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size, 3)
        self.out = nn.Linear(hidden_size, output_size)

    
    def forward(self, input,lengths, hidden):
        input = F.relu(input)
        pack = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hidden = self.gru(pack, hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        final_out = self.out(unpacked)
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
data = arctic_data(torch.Tensor(padded_input), torch.Tensor(padded_output), lengths)
train_loader = DataLoader(dataset = data, batch_size=4, shuffle=True)

loss_fn = torch.nn.MSELoss(size_average=False)

#learning_rate = 1e-4

enc_model = EncoderGRU(hidden_dim,  60) # hidden_dim, input_dim
dec_model = DecoderGRU(hidden_dim, 60) # hidden_size, output size
#model = LSTMpred(hidden, 3, 711, 60) #hidden_dim, hidden_layers, input_size, output_size)
encoder_optimizer = optim.Adam(enc_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
decoder_optimizer = optim.Adam(enc_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

encoder_hidden = enc_model.init_hidden(3,4) # num of layers, batch_size
print("hidden hsape is:", numpy.shape(encoder_hidden))
#print("aftr initilatisatoon", numpy.shape(encoder_hidden))
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum= 0.9)

for i in range(0,30):
  total_loss = 0
  print("epoch", i)
  p =0
  for in_batch, output, lengths in train_loader:
    p = p+1
####    print("am in batch:", p)
    new_lengths = []
    in_batch, output, lengths = sort_batch(in_batch, output, lengths)
#    print("input hsape is:", numpy.shape(in_batch))
    for i in range(0,len(lengths)):
      new_lengths.append(lengths[i])
    enc_output, encoder_hidden = enc_model(autograd.Variable(in_batch, requires_grad = True), new_lengths, encoder_hidden)
   
####    print("encoder output is", numpy.shape(enc_output))

    decoder_hidden = encoder_hidden
    dec_output, decoder_hidden = dec_model(autograd.Variable(output, requires_grad = True), new_lengths, decoder_hidden)

### Calculating the Loss ###########

    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(output), new_lengths, batch_first=True)
    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
####    print("dec output is", numpy.shape(dec_output), "output is", numpy.shape(unpacked))

# write self. init hidden here
# pack the decoder input den unpack it and den perfrom decodong  --> this is because aftr unpacking encoder outputs have the max length in the batch as the seq length
# initilaise hidden for decoder


#    pack = torch.nn.utils.rnn.pack_padded_sequence(autograd.Variable(output), new_lengths, batch_first=True)
#    unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(pack, batch_first=True)
    loss = loss_fn(dec_output, unpacked)
    total_loss += loss.data[0]
    loss.backward(retain_graph=True)
    encoder_optimizer.step()
    decoder_optimizer.step()
# ============ make zero grad for both enc n decoder ========    ####optimizer.zero_grad()
  print("loss is:", total_loss)


