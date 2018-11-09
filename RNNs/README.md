# Steps for data preparation, packing, padding

DATA LOADER

https://www.youtube.com/watch?v=zN49HdDxHi8

1) seq padding --> import torch.nn.utils.rnn as rnn_utils

i) packed = rnn_utils.pack_sequence([a, b, c])
ii) rnn_utils.pad_sequence([a, b, c], batch_first=True)


1. sorting the input, 2. padding, 3. passing through lstm, 4. pack

NOTE:

RNN does not perform computations on padded portions of input --> 
https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099

DATA shuffling before or after packing:

The list just has to be sorted within an individual batch, so you can still shuffle your dataset and randomly sample a batch at a time, 
then sort each batch to send it to pack_padded_sequence.

DATA PROCESSING STEPS

1) data shuffling (along with normlisation) and batching

2) padding and packing of sequences withing the batch

21 march
ToDo:

1) Normalisation --> Need batch normalisation
 i) import torch.nn.functional as F
    x = F.normalize(x, p=2, dim=1)
source : https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/8

2)i) masking to process only non padded seq for loss --> https://discuss.pytorch.org/t/batch-processing-with-variable-length-sequences/3150/3
  ii) New -->  there’s no need for any wrapper. As long as the padded entries don’t contribute to your loss in any way, their 
     gradient will always be 0.
  https://discuss.pytorch.org/t/how-can-i-compute-seq2seq-loss-using-mask/861/5

3) loss on varibale length sequnces --> https://discuss.pytorch.org/t/calculating-loss-on-sequences-with-variable-lengths/9891
4) check requires grad usage

The role of vowel and consonant fundamental frequency, envelope, and temporal fine structure cues to the intelligibility of words and sentences
Statistical analysis of bilingual speaker's speech for cross-language voice conversion.

5) save the model
6) check points

Error found in the network follow this --> 
https://github.com/ngarneau/understanding-pytorch-batching-lstm/blob/master/Understanding%20Pytorch%20Batching.ipynb
