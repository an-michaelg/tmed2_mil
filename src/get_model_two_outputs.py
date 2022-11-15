'''
WARNING: THIS FILE IS INCOMPLETE. PROOF OF CONCEPT THAT ISNT TESTED

# compcars model architecture, featuring resnet encoding layer and attention

import torch
import torch.nn as nn
DIM_EMBEDDING = 128
DIM_HIDDEN = 256
RESOLUTION = 224

def count_module_params(module, title):
    num_parameters = sum([x.nelement() for x in module.parameters()])
    print(f"The number of parameters in the {title}: {num_parameters/1000:9.2f}k")
    
# returns a feature extractor that extends nn.Module
def get_encoder(embedding_dim, pre=True):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=pre)
    num_ftrs = model.classifier.in_features
    # change the final linear layer for transfer learning
    model.classifier = nn.Linear(num_ftrs, embedding_dim)
    #count_module_params(model, 'encoder')
    return model

class GatedAttention(nn.Module): # GA module (https://arxiv.org/pdf/1802.04712.pdf)
    def __init__(self, input_dim, hidden_dim, use_gated=False, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)
        if use_gated:
            self.U = nn.Linear(input_dim, hidden_dim)
        else:
            self.U = None
            
    def forward(self, x): # assume x is Dx1, where D = input_dim 
        hidden = torch.tanh(self.V(x))
        if self.U:
            gate = torch.tanh(self.U(x))
            hidden = hidden * gate # hadamard product
        a = self.w(hidden) # dot product 
        return a
    
class DenseMLP(nn.Module): # simple dense MLP for classification
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        hidden = torch.relu(self.W1(x))
        logits = self.W2(hidden)
        return logits

class GatedAttentionClassifier(nn.Module):
    def __init__(self, embedding_dim, dense_hidden_dim, nc_AS, nc_view, 
                 encoder_pretrained=True, use_gated=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = get_encoder(embedding_dim, encoder_pretrained)
        self.AS_mlp = DenseMLP(embedding_dim, dense_hidden_dim, nc_AS)
        self.view_mlp = DenseMLP(embedding_dim, dense_hidden_dim, nc_view)
        self.attention = GatedAttention(embedding_dim, dense_hidden_dim, use_gated)
    
    def forward(self, x_list):
        # expected to take in an arbitrary length N list of Bx3xHxW
        logits_view, logits_AS, attns = [], [], []
        for x in x_list:
            emb = self.encoder(x) # BxD
            logits_view.append(self.view_mlp(emb)) # BxC
            logits_AS.append(self.AS_mlp(emb)) # BxC 
            attns.append(self.attention(emb)) # Bx1
        
        # normalizing attention
        attns = torch.stack(attns) # NxBx1
        logits_view = torch.stack(logits_view) # NxBxC
        logits_AS = torch.stack(logits_AS) # NxBxC
        return logits_AS, logits_view, attns

class AveragingClassifier(nn.Module):
    def __init__(self, embedding_dim, dense_hidden_dim, nc_AS, nc_view, 
                 encoder_pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.encoder = get_encoder(embedding_dim, encoder_pretrained)
        self.AS_mlp = DenseMLP(embedding_dim, dense_hidden_dim, nc_AS)
        self.view_mlp = DenseMLP(embedding_dim, dense_hidden_dim, nc_view)
    
    def forward(self, x_list):
        # expected to take in an arbitrary length N list of Bx3xHxW
        logits_view, logits_AS = [], []
        for x in x_list:
            emb = self.encoder(x) # BxD
            logits_view.append(self.view_mlp(emb)) # BxC
            logits_AS.append(self.AS_mlp(emb)) # BxC 
        
        logits_view = torch.stack(logits_view) # NxBxC
        logits_AS = torch.stack(logits_AS) # NxBxC
        return logits_AS, logits_view
    
    
def gac_unit_test(use_cuda=True):
    gac = GatedAttentionClassifier(embedding_dim=DIM_EMBEDDING,
                                   dense_hidden_dim=DIM_HIDDEN,
                                   nc_AS=5, nc_view=7,
                                   encoder_pretrained=True,
                                   use_gated=True)
    
    num_parameters = sum([x.nelement() for x in gac.parameters()])
    print(f"The number of parameters in the GAC: {num_parameters/1000:9.2f}k")
    
    batch_size = 4
    list_length = 11
    test_in = [torch.randn((batch_size, 3, RESOLUTION, RESOLUTION)) for _ in range(list_length)]
    if use_cuda:
        if torch.cuda.is_available():
            print('Using GPU acceleration: GPU{}'.format(torch.cuda.current_device()))
            gac = gac.cuda()
            test_in = [k.cuda() for k in test_in]
        else:
            print('CUDA is not available. Using CPU')
    logits_view, logits_AS, attns = gac(test_in)
    return logits_AS, logits_view, attns # should return a NxBxC(1) tensor on CUDA device

#l = gac_unit_test()
        
        
        
        
        
        