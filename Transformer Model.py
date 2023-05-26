#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch')


# In[2]:


get_ipython().system('pip install altair')


# In[3]:


get_ipython().system('pip install torchtext')


# In[4]:


get_ipython().system('pip install spacy')


# In[5]:


get_ipython().system('pip install GPUtil')


# In[6]:


# The provided code imports various modules and packages for this deep learning project


import os   # provides functions for interacting with the operating system.
from os.path import exists   # checks if a file or directory exists at a given path.
import torch   # this imports the PyTorch library, which is a popular deep learning framework.
import torch.nn as nn   # contains various neural network layers and functions.
from torch.nn.functional import log_softmax, pad   # applying log softmax activation and padding operations, respectively.
import math   # provides mathematical functions and constants.
import copy   # provides functions for creating copies of objects.
import time   # provides functions for measuring time and delays.
from torch.optim.lr_scheduler import LambdaLR   # defines a learning rate scheduling strategy.
import pandas as pd   # provides data analysis and manipulation tools.
import altair as alt   # data visualization library.
from torchtext.data.functional import to_map_style_dataset   # converts datasets to the map-style dataset format. 
from torch.utils.data import DataLoader   # creates data loaders that can load data in parallel during training.
from torchtext.vocab import build_vocab_from_iterator   # building a vocabulary from an iterator.
import torchtext.datasets as datasets   #provides access to various datasets commonly used in natural language processing tasks
import spacy   # a natural language processing library.
import GPUtil   # Python GPU monitoring and utilization library.
import warnings   # issuing warning messages.
from torch.utils.data.distributed import DistributedSampler   # sampling data in a distributed training setting.
import torch.distributed as dist   # provides functions and utilities for distributed training. 
import torch.multiprocessing as mp   # multiprocessing support.
from torch.nn.parallel import DistributedDataParallel as DDP   # wraps model with parallel training support in a distributed setting.


# In[7]:


warnings.filterwarnings("ignore")   # prevents warning messages from cluttering the output.
RUN_EXAMPLES = True   # flag or a configuration option. 


# In[8]:


def is_interactive_notebook():   # defines function. 
    
    return __name__ == "__main__"   # if code '__name__' is "__main__" = running main program. Function is true. 
                                    # if code '__name__' is NOT "__main__" = imported code. Function is false. 

# this code allows you to write code that behaves differently depending on whether it is running in an interactive notebook or not.
# can adjust the behavior or execute specific code blocks based on the execution context.
                                     


# In[9]:


# allows to selectively run specific examples or functions when running the script.

def show_example(fn, args=[]):   # executes a specific function 'fn' with optional arguments 'args'.
    
    if __name__ == "__main__" and RUN_EXAMPLES:   # code must be executed as main program not as a module imported by another script.
        return fn(*args)   # '*' before 'args' unpacks the list of arguments, allowing them to be passed individually to the function 'fn'.                           


# In[10]:


# execute a specific example or function only if the script is running as the main program.
def execute_example(fn, args=[]):
    
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


# In[11]:


# placeholder in situations where an optimizer object is required but no actual optimization is needed. 

class DummyOptimizer(torch.optim.Optimizer):   
    
    def __init__(self):   # initialization method of the class. 
        self.param_groups = [{"lr": 0}]   # initialized as a list containing a single dictionary with a key "lr" and a value of 0. Learning rate of 0.
        None
        
    def step (self):   # optimization step of the optimizer.
        None 
        
    def zero_grad(self, set_to_none = False):   # zeros the gradients of the parameters being optimized.
        None


# In[12]:


#  placeholder in situations where a scheduler object is required but no actual scheduling is needed. 

class DummyScheduler:
    def step(self):
        None


# In[13]:


# defines a sequence-to-sequence model with an encoder, decoder, source embedding, target embedding, and a generator. 
# provides methods for performing the forward pass, encoding the source input, and decoding the target input.
# encapsulates the architecture and functionality of an encoder-decoder model for sequence generation tasks.

class EncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):   # parameters are assigned to corresponding attributes of the EncoderDecoder instance using the self keyword.
        
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    # forward pass
    def forward(self, src, tgt, src_mask, tgt_mask):   # calls the decode() method by passing the encoded source (src) and source mask (src_mask) as arguments, along with the target (tgt) and target mask (tgt_mask).
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    
    #  encoding step
    def encode(self, src, src_mask):   # applies the source embedding (src_embed) to the source input (src) and passes the result, along with the source mask (src_mask), to the encoder.
        return self.encoder(self.src_embed(src), src_mask)
    
    # decoding step 
    def decode(self, memory, src_mask, tgt, tgt_mask):   # applies the target embedding (tgt_embed) to the target input (tgt) and passes the result, along with the memory, source mask (src_mask), and target mask (tgt_mask), to the decoder.
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[14]:


# class represents a component of this neural network model.  

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    
    # initializes self.proj as an instance of the nn.Linear module, which performs a linear transformation from d_model to vocab dimensions.
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab) 
    
    # applies the linear transformation (self.proj) to the input x and passes the result through a log softmax activation function using the log_softmax function from torch.nn.functional.
    def forward(self, x):
        return log_softmax(self.proj(x), dim=1)


# In[15]:


# helper function. Creates multiple clones of a given module.
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) 


# In[16]:


# class represents an encoder component of this neural network model.

class Encoder(nn.Module):
    
    # initializes self.layers as a list of N clones of the provided layer.
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)   #  initializes self.norm as an instance of LayerNorm. 
    
    # iterates over each layer in self.layers and applies each layer to the input x along with the provided mask.
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)   # normalized output is returned.
    


# In[17]:


# class represents a layer normalization module. 
# normalizes the activations or outputs of layers in a neural network, helping with the overall stability and convergence of the network during training.

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # initializes self.a_2 as an instance of nn.Parameter with shape features and initialized with ones.
        self.b_2 = nn.Parameter(torch.zeros(features))   # initializes self.b_2 as an instance of nn.Parameter with shape features and initialized with zeros.
        self.eps = eps   # eps is an optional parameter for numerical stability to avoid division by zero.
    
    # computes the mean and standard deviation of 'x'.
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)   # applies layer normalization to x 
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[18]:


# class represents a connection with layer normalization and dropout regularization.
# introduce residual connections and control the impact of the sublayer on the overall network output.

class SublayerConnection(nn.Module):
    
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)   # initializes self.norm as an instance of the LayerNorm class, which performs layer normalization on an input tensor of size size.
        self.dropout = nn.Dropout(dropout)   #  initializes self.dropout as an instance of the nn.Dropout class, which applies dropout regularization with the specified dropout probability.
    
    # applies dropout regularization to the result. 
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# In[19]:


# class represents a single layer of an encoder in this transformer-based model. 
# feed forward network and two sublayers that apply layer normalization, dropout, and residual connections.
# during the forward pass, the input x undergoes self-attention followed by a feed-forward operation, with layer normalization and residual connections applied between them.

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward   # initializes 'self.self_attn' as an instance of a self-attention mechanism, 'self.feed_forward' as an instance of a feed-forward network. 
        self.sublayer = clones(SublayerConnection(size, dropout), 2)   # 'self.sublayer' as a list of two 'SublayerConnection' instances.
        self.size = size
        
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))   # 'lambda' function that applies the self_attn mechanism to x (performing self-attention on x using x as queries, keys, and values) with the provided mask.
        return self.sublayer[1](x, self.feed_forward)   # output of the second sublayer is returned. 


# In[20]:


# class represents a decoder component of this neural network model. 

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    
    # initializes self.layers as a list of N clones of the provided layer.
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    # iterates over each layer in self.layers and applies each layer to the input x, memory, src_mask, and tgt_mask. 
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)   # applies a normalization operation (self.norm) to the output x.
        return self.norm(x)


# In[21]:


# class represents a single layer of a decoder in this neural network model.
# consists of self-attention, source attention, and feed-forward components, along with layer normalization, dropout, and residual connections.

class DecoderLayer(nn.Module):
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size   # input tensor. 
        self.self_attn = self_attn   # instance of a self-attention mechanism. 
        self.src_attn = src_attn    # instance of a source attention mechanism. 
        self.feed_forward = feed_forward   # instance of a feed-forward network.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)   # applies layer normalization, dropout, and a sublayer function to the input tensor.
    
    # input x undergoes self-attention, followed by source attention and feed-forward operations.
    def forward(self, x, memory, src_mask, tgt_mask):
        
        m = memory 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   # layer normalization, dropout, and residual connections are applied between each sublayer.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# In[22]:


# generates a mask matrix for attending to future positions in a sequence.

def subsequent_mask(size):   # attention is applied to all positions in the sequence except the future positions.
    attn_shape = (1, size, size)    # defines the shape of the attention mask tensor as a tuple. Shape represents a 3-dimensional tensor with a batch size of 1 and dimensions size x size.
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).type(    # creates an upper triangular matrix with ones above the main diagonal and zeros on and below the main diagonal. 
        torch.uint8                                                             # 'diagonal' argument is set to 1, which means the main diagonal and elements below it are excluded from the upper triangular part.
    )
    return subsequent_mask == 0   # returns a boolean mask where the values that are equal to 0 in the subsequent_mask tensor are set to True, and the values that are not equal to 0 are set to False. 
                                  # allowing the attention mechanism to attend to only the previous or current positions.


# In[23]:


# generates a visualization of the subsequent mask using the altair library. 
# visualization allows the user to explore and understand the pattern of the subsequent mask.

def example_mask():    
    LS_data = pd.concat(   # concatenates a list of data frames that contain the values for the "Subsequent Mask", "Window", and "Masking" columns. 
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),   # flattening the subsequent mask tensor generated by the subsequent_mask function.
                    "Window": y,   # set based on the iteration over the ranges of 20 for both variables.
                    "Masking": x,   
                }
        )
            for y in range (20)
                for x in range(20)
            ]
        )
    return (
            alt.Chart(LS_data)   # visualization is created. 
            .mark_rect()   # chart type is rectangular mark. 
            .properties(height = 250, width = 250)   
            .encode(
                alt.X("Window:0"),   # setting the "Window" and "Masking" variables as the X and Y axes. 
                alt.Y("Masking:0"),
                alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),   # color scale is set to "viridis".
            )
            .interactive()   # enables interactivity for the chart.
        )
    
    show_example(example_mask)


# In[24]:


# computes the scaled dot product attention mechanism. 

def attention(query, key, value, mask=None, dropout=None): # tensors. "none" makes it optional. 
    "Compute 'Scaled Dot Production Attention'"
    d_k = query.size(-1)   # determines the dimensionality of the query tensor by getting the size of the last dimension. 
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   # computes the dot product of the query and key tensors using torch.matmul, and divides it by the square root of d_k to scale the dot product.
                                                                           # 'key.transpose(-2, -1)' aligns the dimensions for the matrix multiplication.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # replacing the corresponding entries with a large negative value (-1e9) ensuring they are not attended to in the subsequent softmax computation.
        p_attn = scores.softmax(dim = -1)   # applies the softmax function along the last dimension of the scores tensor, resulting in attention weights (p_attn) that sum up to 1. 
                                            # computes the normalized attention distribution.
            
        if dropout is not None:
            p_attn = dropout(p_attn)   # applies dropout to the attention weights by passing them through the dropout module. 
                                       # dropout randomly sets elements of the tensor to zero, which helps prevent overfitting and encourages robustness.
                
        return torch.matmul(p_attn, value), p_attn   # returns the weighted sum of the value tensor based on the attention weights ('p_attn') by matrix multiplication of 'p_attn' and value. 


# In[25]:


# class implements the multi-headed attention mechanism.

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0   # checks if the 'd_model' is divisible by 'h' without any remainder. Ensures that the dimensionality can be evenly divided among the attention heads.
        self.d_k = d_model // h   # computes the dimensionality of each attention head ('d_k') by dividing 'd_model' by the number of attention heads 'h'.
        self.h = h   # assigns the number of attention heads to the attribute. 
        self.linears = clones(nn.Linear(d_model, d_model), 4)   # creates a list of 4 linear transformation layers. Used to transform the query, key, and value tensors for each attention head.
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)   # initializes the dropout module with the specified dropout probability.
    
    # implements the forward pass of the multi-headed attention mechanism.
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # adds an extra dimension to the mask tensor to align with the attention dimensions.
        nbatches = query.size(0)   # determines the number of batches by getting the size of the first dimension of the query tensor.
            
        query, key, value = [   # applies linear transformations to the query, key, and value tensors using the self.linears list of linear layers.
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)   # The input tensors are passed through the linear layers, and the resulting tensors are reshaped to have dimensions (nbatches, -1, self.h, self.d_k) and transposed along the second and third dimensions.
                
            for lin, x in zip(self.linears, (query, key, value))
        ]
            
        x, self.attn = attention(   # calls the 'attention' function  with the transformed query, key, and value tensors, as well as the mask and dropout. Computes the attention weights and applies them to the value tensor.
            query, key, value, mask=mask, dropout=self.dropout
        )
            
        x = (    # reshapes the tensor x by transposing it along the second and third dimensions
            x.transpose(1,2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # deletes the variables query, key, and value to free up memory.
        del query
        del key 
        del value
            
        return self.linears[-1](x)   # applies a linear transformation to the reshaped tensor x using the last linear layer in `self. 


# In[26]:


# implements the feed-forward neural network (FFN) component in this transformer model. 

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)   # creates a linear transformation layer (nn.Linear) with input dimension d_model and output dimension d_ff. This layer is responsible for the first linear transformation. 
        self.w_2 = nn.Linear(d_ff, d_model)   # creates another linear transformation layer with input dimension d_ff and output dimension d_model. This layer is responsible for the second linear transformation. 
        self.dropout = nn.Dropout(dropout)   # initializes the dropout module with the specified dropout probability.
        
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))   # applies the first linear transformation to the input tensor x, resulting in an intermediate tensor.
                                                            # '.relu()' applies the ReLU activation function to the intermediate tensor element-wise.
                                                            # then applies dropout to the ReLU-activated tensor.
                                                            # lastly applies the second linear transformation to the dropout-applied tensor, producing the final output tensor of the FFN.


# In[27]:


# encapsulates the functionality of input embeddings in this transformer model.
# maps token indices to embedding vectors and scales the vectors to control their magnitudes.

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)   # creates an embedding layer (nn.Embedding) with vocab as the number of unique tokens and d_model as the dimensionality of the embedding vectors.
        self.d_model = d_model                    # used to map token indices to their corresponding embedding vectors.
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # looks up the embedding vectors for the given token indices x using the embedding layer 'self.lut'.
                                                      # then computes the square root of self.d_model, which is used for scaling the embedding vectors.
                                                      # scales the embedding vectors to prevent their magnitudes from getting too large or too small.
         


# In[28]:


# implements the positional encoding for the input sequence in this transformer model.
# creates a positional encoding tensor based on the positions and dimensions of the input elements and adds this encoding to the input tensor.
# positional encoding helps the transformer model capture the sequential order of the input sequence.

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)   # creates a dropout layer with the given dropout probability.
        
        pe = torch.zeros(max_len, d_model)   # creates a tensor of shape filled with zeros to store the positional encoding.
        position = torch.arange(0, max_len).unsqueeze(1)   # creates a tensor of shape ('max_len, 1') containing values from 0 to 'max_len-1' to represent the positions of the sequence elements.
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)   # computes a tensor of shape ('d_model/2') containing the values for the division term in the positional encoding formula.
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # computes the sine values for the first 10 dimensions of the positional encoding using the formula. 
        pe[:, 1::2] = torch.cos(position * div_term)   # computes the cosine values for the remaining dimensions of the positional encoding using the formula.
        pe = pe.unsqueeze(0)   # adds a new dimension at the beginning of the tensor to match the batch dimension.
        self.register_buffer("pe", pe)   # registers the positional encoding tensor 'pe' as a buffer in the model. 
                                         # buffers are persistent tensors that are not considered model parameters but are part of the model's state.
        
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)   # adds the positional encoding tensor 'self.pe' to the input tensor 'x'. The positional encoding is added up to the length of the input sequence.
                                                                # ensures that the positional encoding is not part of the computational graph for gradient computation.
        return self.dropout(x)


# In[29]:


# applies positional encoding to a tensor of zeros and visualizing the resulting embeddings across different dimensions and positions. 
# provides a visual representation of how positional encoding affects the embedding values in this transformer model.

def example_positional():
    pe = PositionalEncoding(20, 0)   # creates an instance of the class with a dimensionality of 20 and dropout probability of 0.
    y = pe.forward(torch.zeros(1, 100, 20))   # applies the positional encoding to a tensor of zeros with shape (1, 100, 20). Applies the positional encoding to the input tensor and returns the encoded tensor 'y'.
    
    data = pd.concat(   # creates a Pandas DataFrame called 'data' by concatenating multiple DataFrames.
        [               
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            
            for dim in [4,5,6,7]   # each DataFrame represents the embedding values of a specific dimension ('4, 5, 6, 7') in the encoded tensor 'y'. 
                                   # contains columns for the embedding values, the corresponding position index, and the dimension.
        ]
    )

    
    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )  
    

show_example(example_positional)   # display the chart generated. 


# In[30]:


# sequence-to-sequence model for machine translation using the provided hyperparameters and initializes its parameters.

def make_model(                 
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1 
):  # size of source vocab, size of target vocab, number of encoder/decoder layers, dimensionality of model, dimensionality of feed-forward layer, number of attention heads, dropout rate.  
    
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab), 
    )
    
    # initialize parameters with glorot / fan_avg 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# In[31]:


# generate predictions by encoding a source sequence and iteratively decoding the output sequence step by step.
# output sequence is built by selecting the token with the highest probability at each decoding step.
def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()   # evaluation mode (disables any training-specific operations like dropout). 
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])   # each integer represents a token in the source language.
    src_mask = torch.ones(1, 1, 10)   # indicates which elements in the source sequence should be attended to during the encoding process.
    
    memory = test_model.encode(src, src_mask)   # obtain the encoded representation of the source sequence.
    ys = torch.zeros(1, 1).type_as(src)   # 'ys' initialized as a tensor of zeros with shape (1,1). 
    
    for i in range(9):
        out = test_model.decode(    # ensures that the model attends only to the previously generated tokens and not the future ones. 
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])    # obtain the probability distribution over the target vocabulary.
        _, next_word = torch.max(prob, dim=1)    # index of the token with the highest probability, representing the predicted next word.
        next_word = next_word.data[0]    
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1    # concatenating a tensor of shape (1,1) filled with the predicted word index.
        )
    print("Example Untrained Model Prediction:", ys)   # predicted output sequence, which represents an example of the model's prediction without being trained.


# In[32]:


def run_tests():
    for _ in range(10):   # calls function 10x
        inference_test()    # allows for running the inference process multiple times and obtaining different predictions for each iteration.
        
        
show_example(run_tests)


# In[33]:


# creates batches of data for training or inference in a sequence-to-sequence model.
# source and target sequences as input and generates source masks, target masks, and modified target sequences for training.

class Batch:
    
    def __init__(self, src, tgt=None, pad=2): # source sequence tensor, target sequence tensor, padding value
        self.src = src 
        self.src_mask = (src != pad).unsqueeze(-2)    # comparing the source tensor with the padding value. The result is a binary mask tensor with True values where the source sequence is not padded and False values where it is padded. The mask is unsqueezed along a new dimension -2 to match the expected shape for subsequent computations.
        
        if tgt is not None:
            self.tgt = tgt[:, :-1] 
            self.tgt_y = tgt[:, 1:]    # This represents the expected output for the model, shifting by one position.
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()    # calculated by counting the number of non-padding tokens. 
            
    @staticmethod
    def make_std_mask(tgt, pad):   # creates a mask to hide padding and future words in the target sequence. It uses the target tensor and the padding value to create the mask.
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(    # creates a mask to hide future words, ensuring that the model attends only to the previously generated tokens and not the future ones.
            tgt_mask.data
        )
        return tgt_mask


# In[34]:


# store and track the training-related state and statistics during the training process. 

class TrainState:
    
    step: int = 0   # current training step. Initialized to 0.
    accum_step: int = 0   #  accumulated training steps.
    samples: int = 0   # total number of training samples processed so far.
    tokens: int = 0   #  total number of tokens processed so far (track progress or measure the training efficiency.)


# In[35]:


# single epoch of training or evaluation on a given dataset using a sequence-to-sequence model.
# building block for implementing the training loop or evaluation loop. 
# logic for processing batches, computing loss, performing optimization steps, and tracking training statistics.

def run_epoch(
    data_iter,   # iterates batches of data from the dataset.
    model,   # used for training or evaluation (sequence-to-sequence).
    loss_compute,   # computes the loss and any associated metrics.
    optimizer,   # responsible for updating the model's parameters during training.
    scheduler,   # adjusts the learning rate during training. 
    mode="train",   # string indicating the mode of operation to "train" or "train+log". 
    accum_iter=1,   # number of gradient accumulation steps before performing a backward pass and optimization (default is 1).
    train_state=TrainState(),   # class that stores the training-related state and statistics. 
):
    start = time.time()   # timestamp to measure the elapsed time during the epoch. 
    total_tokens = 0   # running total of the number of tokens processed during the epoch.
    total_loss = 0   # running total of the loss computed during the epoch. 
    tokens = 0   # number of tokens processed since the last log output.
    n_accum = 0   # number of accumulated steps performed so far.
    
    for i, batch in enumerate(data_iter):
        out = model.forward(    # obtain the model's output.
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        
        if mode == "train" or mode == "train+log":
            loss_node.backward()   # loss is backpropagated through the computational graph to compute gradients.
            train_state.step += 1  # incremented by 1 to track the training step.
            train_state.samples += batch.src.shape[0]   # objects are updated.
            train_state.tokens += batch.ntokens
            
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)   # update model's parameters, gradients are zeroed. 
                n_accum += 1   # number of accumulated steps is incremented. 
                train_state.accum_step += 1 
                scheduler.step()
                
            total_loss += loss   # loss and number of tokens in the batch are added to the running totals. 
            total_tokens += batch.ntokens
            tokens += batch.ntokens   # number of tokens processed since the last log output is incremented by the number of tokens in the current batch. 
            if i % 40 == 1 and (mode == "train" or mode == "train+log"):
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start 
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                        % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                    )
            start = time.time()
            tokens = 0 
                    
            del loss 
            del loss_node
            
    if total_tokens == 0:   #  function returns infinity loss
        return float("inf"), train_state

    return total_loss / total_tokens, train_state   # overall loss for the entire epoch.        


# In[36]:


# calculates the learning rate for a specific training step
# provides a strategy for gradually increasing the learning rate during the warm-up phase and decaying it afterwards to optimize the training process.

def rate(step, model_size, factor, warmup):  # scaling factor applied to the learning rate & number of warm-up steps during which the learning rate gradually increases.
    
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))   # scales the learning rate based on the inverse square root of the current step. Initially, the learning rate decreases quickly, and later it decreases more slowly.
    )                                                                         # scales the learning rate based on the warm-up phase. During the warm-up phase, the learning rate gradually increases with the step number to stabilize the training process.


# In[37]:


# demonstrates a learning rate schedule for different combinations of model size and warm-up steps.

def example_learning_schedule():
    opts = [   # different combinations. 
        [512, 1, 4000],
        [512, 1, 8000],
        [256, 1, 4000], 
    ]
    
    dummy_model = torch.nn.Linear(1, 1)    # placeholder model. 
    learning_rates = []   # empty list. 
    
    for idx, example in enumerate(opts):   # index and value of each combination.
        
        optimizer = torch.optim.Adam(   # Adam optimizer is created with a learning rate of 1, betas=(0.9, 0.98), and epsilon of 1e-9. 
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)   # calculates the learning rate. 
        )
        tmp = []   # temporary list stores the learning rate values at each training step.
        
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()    # update the model's parameters.
            lr_scheduler.step()  # update the learning rate based on the defined schedule.
        learning_rates.append(tmp)   # tmp list containing the learning rate values is appended to the learning_rates list.
        
    learning_rates = torch.tensor(learning_rates)
    
    alt.data_transformers.disable_max_rows()  # visualize the learning rate schedule using Altair library. 
    
    opts_data = pd.concat( # DataFrame is created by concatenating individual DataFrames for each warm-up index. Each DataFrame contains the learning rate values, model size, warm-up combination, and training step.
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :], 
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )
# x-axis representing the training step, y-axis representing the learning rate, and different colors representing different model size:warm-up combinations.    
    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size\:warmup:N")
        .interactive()
    )  

example_learning_schedule()


# In[38]:


# regularization technique helps prevent overfitting and encourages the model to be more robust by reducing the model's confidence in its predictions.

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")   # (Kullback-Leibler Divergence Loss) with reduction set to "sum". This will be used as the loss function for calculating the label smoothing loss.
        self.padding_idx = padding_idx   # padding index.
        self.confidence = 1.0 - smoothing   # float representing the confidence value assigned to the target class
        self.smoothing = smoothing   # float representing the smoothing factor, which determines the amount of smoothing applied to the labels.
        self.size = size   # size of the output space.
        self.true_dist = None   # tensor that will store the true distribution after applying label smoothing.
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)   # update the relevant indices.
        true_dist[:, self.padding_idx] = 0   # ignore it during the loss calculation. 
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# In[39]:


# visualizes the target distribution after applying label smoothing.

def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict=torch.FloatTensor(   # predicted probabilities for each class. It is a 5x5 tensor with values specified in the code.
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))   # applies the logarithm to the predicted probabilities.
    data = []
    for y in range(5):
        for x in range(5):
            data.append({
                "target distribution": crit.true_dist[x, y].flatten().tolist(),   # flattened target distribution after applying label smoothing. 
                "columns": y,
                "rows": x,
            })
    LS_data = pd.DataFrame(data)    # list of dictionaries. 

    return (
        alt.Chart(LS_data)   # visualize the target distribution. 
        .mark_rect(color="Blue", opacity=1)   # represent the target distribution values. 
        .properties(height=200, width=200)   # row index is mapped to the y-axis.
        .encode(
            alt.X("columns:O", title=None),   # column index is mapped to the x-axis.
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )
show_example(example_label_smoothing)  # render the Altair chart showing the target distribution after applying label smoothing. 


# In[40]:


#  calculates the loss

def loss(x, crit):  # input value & instance of a criterion
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])   # predicted probabilities for each class and has the shape 1x5.
    return crit(predict.log(), torch.LongTensor([1])).data   # logarithm of the predicted probabilities. Data resulting loss is accessed and returned.


# In[41]:


# visualizes the loss values for different input values using a criterion. 
# chart provides insights into how the loss changes with varying input values, allowing for the analysis of penalization effects.

def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1,100)], # loss values calculated 1-99. 
            "Steps": list(range(99)),   # corresponding steps or input values from 0-98. 
        }
    ).astype("float")
    
    return (
        alt.Chart(loss_data)
        .mark_line()   # chart is configured to display a line. 
        .properties(width=350)
        .encode(
            x="Steps",   # x-axis is specified as "Steps". 
            y="Loss",    # y-axis is specified as "Loss". 
        )
        .interactive()
    )
                
show_example(penalization_visualization)


# In[46]:


# generates random data for a source-target copy task. 

def data_gen(V, batch_size, nbatches):   # vocab size, size of each batch, & number of batches to generate.
    
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()   # ensure tensors are detached from the computation graph and do not require gradients.
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)   # allow the caller to iterate over the generated batches of data.


# In[43]:


# simple loss computation and training function.

class SimpleLossCompute:
    "A simple loss compute and train funtion."
    
    def __init__(self, generator, criterion):  
        self.generator = generator   # maps the input x to the output tensor
        self.criterion = criterion   # compute the loss between the predicted output x and the target y. 
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(   # loss between the flattened output and target tensors
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)   # x and the target tensor y are reshaped to be contiguous and flattened. 
            )
            / norm  # computed loss is divided by the normalization factor. 
        )
        return sloss.data * norm, sloss   # computed loss is then multiplied by the normalization factor & returned as the first value.


# In[44]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    print("Memory shape:", memory.shape) # print the memory tensor shape
    
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    print("Ys shape:", ys.shape) # print ys tensor shape
    
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # print("out shape:", out.shape) # print out Tensor shape
        
        prob = model.generator(out[:, -1])
        # print("prob shape:", prob.shape) # print out prob tensor shape
        
        _, next_word = torch.max(prob, dim=1)
        # print("next_word shape:", next_word.shape) # print next_word tensor shape
        
        next_word = next_word[0].item()
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# In[45]:


# simple model training process

def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)   # loss function that applies label smoothing to the model's output probabilities during training.
    model = make_model(V, V, N=2)   # create a model.
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,   # update the parameters of the model during training.  
        lr_lambda=lambda step: rate(   # learning rate scheduler. Adjusts the learning rate of the optimizer based on the current training step.
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )
    
    batch_size = 80
    for epoch in range(20):   # model is trained and evaluated.
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),   # loss computation function.
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",   # evaluation result retrieved. 
        )[0]
        
    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])   # source sequence tensor.
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)   # maximum sequence length. Source mask tensor. 
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))   # decodes the source sequence using a greedy approach.
    
execute_example(example_simple_model)

