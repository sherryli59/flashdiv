import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FourierFeatures(nn.Module):
    def __init__(self, dim, sigma=2, period=1):
        super().__init__()
        self.dim = dim
        self.sigma = sigma
        self.period = period

    def forward(self,x):
        """
        Create sinusoidal embeddings.
        :param x: an (B, N, d) Tensor of positions (batch,particle,dim).
        :param dim: the dimension of the output.
        :param sigma: controls the scale of frequency.
        :return: an (B, N, D) Tensor of positional embeddings.
        """
        d = x.shape[-1]
        nfreqs = self.dim // (2*d)
        freqs = torch.pow(self.sigma, torch.arange(start=0, end=nfreqs, dtype=torch.float32) 
        ).to(device=x.device)
        args = x[:,:, None].float() * freqs[None,None,:,None] *2* math.pi / self.period
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).flatten(start_dim=2)
        if self.dim % (2*d)>0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,self.dim % (2*d)])], dim=-1)
        return embedding

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_input, d_k, d_v, dropout=0, residual=True):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_input, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_input, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_input, bias=False)
        self.layer_norm = nn.LayerNorm(d_input,eps=1e-6)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, qkv, mask=None):
        q = k = v = qkv
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        if self.residual:
            q += residual
            q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0, residual=True):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):

        residual = x

        x = self.w_2(F.gelu(self.w_1(x),approximate="tanh"))
        x = self.dropout(x)

        if self.residual:
            x += residual
            x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_input, d_hidden, n_head, d_k, d_v, dropout=0, ada_ln=True,is_final_layer=False):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_input, d_k, d_v, dropout=dropout,residual=False)
        self.pos_ffn = PositionwiseFeedForward(d_input, d_hidden, dropout=dropout,residual=False)
        self.norm1 = nn.LayerNorm(d_input,elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_input,elementwise_affine=False, eps=1e-6)
        if ada_ln: #use adaptive layer normalization
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),nn.Linear(d_input, 6 * d_input, bias=True))
        self.is_final_layer = is_final_layer

    def forward(self, x , context=None, slf_attn_mask=None):
        if hasattr(self,'adaLN_modulation') and context is not None:
            shift_attn, scale_attn, gate_attn, shift_pff, scale_pff, gate_pff = self.adaLN_modulation(context).chunk(6, dim=1)
        else:
            shift_attn = scale_attn = gate_attn = shift_pff = scale_pff = gate_pff = torch.zeros_like(x[:,0])
        x_modulated = modulate(self.norm1(x),shift=shift_attn, scale=scale_attn)

        if not(self.is_final_layer):
            #self-attention
            attn_output, attn = self.slf_attn(
                x_modulated, mask=slf_attn_mask)      
            x = x+gate_attn.unsqueeze(1)*attn_output  #residual connection
            # postionwise feed forward
            pff_output = self.pos_ffn(modulate(self.norm2(x),shift=shift_pff, scale=scale_pff))
            x = x+gate_pff.unsqueeze(1) * pff_output #residual connection
            return x, attn

        else:
            return x_modulated, []


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_input, d_hidden, n_head, d_k, d_v, dropout=0):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_input, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_input, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_input, d_hidden, dropout=dropout)

    def forward(
            self, dec_input, x,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, x, x, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

class MLP(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class Transformer(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_input, d_output, n_layers=4, n_head=4, d_k=128, d_v=128,
            d_hidden=128, dropout=0, time_emb=True, period=5.4):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_hidden, d_hidden, n_head, d_k, d_v, dropout=dropout, is_final_layer=(i==n_layers))
            for i in range(n_layers+1)]) #include one additional normalization layer
        self.fourier_features = FourierFeatures(d_hidden,period)
        self.affine_in = nn.Linear(d_input,d_hidden)
        #print device
        self.affine_out = nn.Linear(d_hidden,d_output)
        self.d_input = d_input
        #self.norm = nn.BatchNorm2d(1,affine=True)
        if time_emb:
            self.time_emb = TimestepEmbedder(hidden_size=d_hidden)
        #self.norm = nn.LayerNorm(d_input)
        self.initialize_weights()

    def forward(self, input, t=None, mask=None, return_attns=False):

        enc_slf_attn_list = []
        # -- Forward
        #x = self.fourier_features(x)
        x = self.affine_in(input)
        x = self.dropout(x)
        if t is not None:
            if t.dim()==0:
                t = t*torch.ones(x.shape[0]).to(x.device)
            time_emb = self.time_emb(t)
        else:
            time_emb = None

        for enc_layer in self.layer_stack:
            x, enc_slf_attn = enc_layer(x, context=time_emb, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        x = self.affine_out(x)
        if return_attns:
            return x, enc_slf_attn_list
        return x


    def initialize_weights(self):
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            if hasattr(self,'time_emb'):
                # Initialize timestep embedding MLP:
                nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
                nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            for layer in self.layer_stack:
                nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.affine_out.weight, 0)
            nn.init.constant_(self.affine_out.bias, 0)

class TransformerForceField(Transformer):
    ''' Transformer modified to represent a conservative force field. '''

    def __init__(self, d_input, n_layers=4, n_head=4, d_k=128, d_v=128,
                 d_hidden=128, dropout=0, time_emb=True, period=5.4):
        super().__init__(d_input, 1, n_layers, n_head, d_k, d_v, d_hidden, dropout, time_emb, period)  # Output scalar potential

    def forward(self, input, t=None, mask=None, return_attns=False):
        # Compute gradient of the potential
        input = input.detach().requires_grad_(True)
        with torch.enable_grad():
            # Compute potential energy (scalar output)
            potential = super().forward(input, t, mask, return_attns).squeeze(-1)
            force = -torch.autograd.grad(
                outputs=potential.sum(),  # Sum to compute gradient w.r.t. the input
                inputs=input,
                create_graph=True
            )[0]
        return force  # Return both force and potential

if __name__=="__main__":
    model = Transformer(2,3)
    x = torch.randn(5,10,2)
    x_flipped = torch.flip(x,dims=[1])
    t1 = torch.randn(5)
    t2 = torch.randn(5)
    output1 = model(x,t1)
    output2 = torch.flip(model(x_flipped,t1),dims=[1])
    print(output1-output2)
