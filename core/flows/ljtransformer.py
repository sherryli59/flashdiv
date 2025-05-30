from einops import rearrange, repeat, reduce, einsum
import torch
import torch.nn as nn
from .architectures import FlowNet
import torch.nn.functional as F
from torch.func import vmap, jacrev, functional_call


class TransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim, key_dim,  nbparticles):
        super().__init__()
        self.input_dim = input_dim
        self.nbparticles = nbparticles
        self.embed_dim = embed_dim
        self.key_dim = key_dim

        # we have multipe encoders to avoid the symmetry issue.
        self.encoders = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(self.input_dim + 1 , self.embed_dim),
            # nn.ReLU(),
            # nn.Linear(self.embed_dim, self.embed_dim)
            ) for _ in range(self.nbparticles)])

        self.decoder = nn.Linear(self.embed_dim, input_dim) # I guess this one can be shared for now and kept as a linear layer --> Note that if we have some MLP that's another call to jacobian
        self.Q = nn.Linear(self.embed_dim, self.key_dim)
        self.K = nn.Linear(self.embed_dim, self.key_dim)
        self.V = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size, nbpart, dim)
        """


        repeat_t = repeat(t, 'b 1 -> b p 1', p=self.nbparticles)
        xt = torch.cat((x.clone(), repeat_t), dim=2)


        # reshape
        xt = rearrange(xt, 'b p d  -> p b d')

        xt_encoded = torch.zeros(xt.shape[0], xt.shape[1], self.embed_dim).to(xt)

        for k,seq_encoded in enumerate(xt):
            xt_encoded[k] = self.encoders[k](seq_encoded)
        # reshape
        xt_encoded = rearrange(xt_encoded, 'l b d -> (b l) d')

        query = rearrange(
            self.Q(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        key = rearrange(
            self.K(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        value = rearrange(
            self.V(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        scaled_dot_product = F.scaled_dot_product_attention(query, key, value)

        # print(scaled_dot_product.shape)
        out = rearrange(
            self.decoder(
                rearrange(scaled_dot_product, 'b l d -> (b l) d')
            ),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        return out

    ### useful functions for the divergence

    #calling jacobian encoders over the batch
    def make_functionnalized_encoders(self):

        functionnalized_encoders = []
        encoder_params = []
        for e in self.encoders:

            # model = paralleltransformernet.encoders[0]
            params = dict(e.named_parameters())

            def fmodel(params, inputs): #functional version of model
                return functional_call(e, params, inputs)
            functionnalized_encoders.append(fmodel)
            encoder_params.append(params)

        self.functionnalized_encoders = functionnalized_encoders
        self.encoder_params = encoder_params

    ## These are the functions we do not want to loose !!!

    # vectorized version of softmax gradf
    def batchedgradsoftmax(s):
        """
        s : (b, v) batch and vector size

        returns
            grad : (b, part, part, part)
        """
        es = torch.exp(s)
        v = s.shape[-1]
        sumes = repeat(
            reduce(
                es, 'b v -> b', 'sum'),
                'b -> b r1 r2',
                r1=v, r2=v)
        div = repeat(es, 'b v -> b v r', r=s.shape[-1])
        div = - div * rearrange(div, 'b v r -> b r v')
        div /= sumes ** 2
        div +=torch.stack(
            [torch.diag(e, 0) for e in es]) / sumes
        return div

    # linear part
    def batchedlineargrad(self, keys, query):
        """
        keys : (b, p, keydim)
        self : transformer

        returns
        grad : (b, part, part, embedim)
        """
        Q = self.Q.weight # assuming no bias
        K = self.K.weight

        r_k = repeat(keys, 'b p d -> b r p d', r=keys.shape[1])
        div = r_k @ Q
        diag_terms = query @ K #(b, p, embeddim)
        iden = torch.eye(diag_terms.shape[1]).to(diag_terms)
        diag_terms = torch.einsum('ij,bik->bijk', iden, diag_terms)

        return div + diag_terms

    #encoder jacobians
    def encoder_jacobian(self, x,t):
        """
        x : (b, p, dim)
        t : (b, 1)
        self : transformer

        returns
        jacobian : (b, part, embdedim, 2)
        """

        all_jacs = []
        k=0
        for model,params in zip(self.functionnalized_encoders, self.encoder_params): # need to define these

            inputs = torch.cat((xtest[:,k],t), dim=1)

            result = vmap(jacrev(model, argnums=(1)), in_dims=(None,0))(params, inputs)[:,:,:-1]    #(b, embdedim, 2)
            all_jacs.append(result)
            k+=1

        all_jacs = rearrange(all_jacs, 'p b embeddim d -> b p embeddim d')

        return all_jacs

    @torch.no_grad()
    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size)
        """

        repeat_t = repeat(t, 'b 1 -> b p 1', p=self.nbparticles)
        xt = torch.cat((x.clone(), repeat_t), dim=-1)
        xt = rearrange(xt, 'b p d  -> p b d')
        xt_encoded = torch.zeros(xt.shape[0], xt.shape[1], self.embed_dim).to(xt)

        for k,seq_encoded in enumerate(xt):
            xt_encoded[k] = self.encoders[k](seq_encoded)

        xt_encoded = rearrange(xt_encoded, 'l b d -> (b l) d')

        # precompute query, key and values
        # reshape

        query = rearrange(
            self.Q(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        key = rearrange(
            self.K(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        value = rearrange(
            self.V(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        # decompose the dotattention operation because we will need these matrices after
        r_q = repeat(query, 'b p d -> b p r d', r=self.nbparticles)
        r_k = repeat(key, 'b p d -> b r p d', r=self.nbparticles)
        r_v = repeat(value, 'b p d -> b r p d', r=self.nbparticles)
        scale = 1 / r_q.shape[-1] ** 0.5
        dot_prod = (r_q * r_k).sum(-1) * scale # this is litterally the attention matrix
        sm = F.softmax(dot_prod, dim=-1) # (b, p, r) # will ned it for the last jacobian.

        # we need three gradients here, the softmax part, the linear part and the encoder part

        # jac viz softmax
        smgrad = rearrange(
            batchedgradsoftmax(
                rearrange(dot_prod, 'b p r -> (b p) r ')),
            '(b p) r d-> b p r d',
            p=dot_prod.shape[1])


        #jac viz linear layers
        lgrad = batchedlineargrad(self, key, query) * scale # I believe we got to scale here because it's part of the linear bit

        # jac viz encoder
        encoder_jac = encoder_jacobian(self, xtest,ttest) #(b p embeddim input_dim)

        # matmuliply all of then using matmuls
        final_jac = (smgrad @ (lgrad @ encoder_jac))

        #decode values --> works because decoder is linear
        decoded_values = rearrange(
            self.decoder(
                rearrange(value, 'b p d -> (b p) d')
            ),
            '(b p) d -> b p d',
            p=self.nbparticles)

        expand_decoded_values = repeat(
            decoded_values,
            'b p d -> b r1 p d ',
            r1=self.nbparticles,
            # r2=pflow.input_dim,
        )

        # this is a hack becasue for each decoded vector we only care about the derivative viz the corresponding coordinate
        div_values = expand_decoded_values * final_jac #(b p p input_dim)

        # we sum over everything but the batch dimension
        div = reduce(div_values, 'b p r d -> b', 'sum')

        # last bit is only the diagonal terms

        div_2 = (self.decoder.weight @ self.V.weight)  @ encoder_jac # (b, p, d, d)
        smdiag = torch.einsum('bpp->bp', sm)
        # print(div_2.shape)
        div_2 = torch.einsum('bpii->bp', div_2)
        div_2 = reduce((div_2 * smdiag), 'b p -> b', 'sum')

        return div+div_2


    def instantiate(self):
            return TransformerFlowLJ(
                self.input_dim,
                self.embed_dim,
                self.key_dim,
                self.nbparticles
            )

class EquivariantTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim, key_dim,  nbparticles):
        super().__init__()
        self.input_dim = input_dim
        self.nbparticles = nbparticles
        self.embed_dim = embed_dim
        self.key_dim = key_dim

        # we have multipe encoders to avoid the symmetry issue.
        self.encoder =  nn.Linear(self.input_dim + 1, self.embed_dim)
        self.decoder = nn.Linear(self.embed_dim, input_dim) # I guess this one can be shared for now and kept as a linear layer --> Note that if we have some MLP that's another call to jacobian
        self.Q = nn.Linear(self.embed_dim, self.key_dim)
        self.K = nn.Linear(self.embed_dim, self.key_dim)
        self.V = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size, nbpart, dim)
        """


        repeat_t = repeat(t, 'b 1 -> b p 1', p=self.nbparticles)
        xt = torch.cat((x.clone(), repeat_t), dim=2)


        # reshape
        xt = rearrange(xt, 'b p d  -> p b d')
        xt_encoded = self.encoder(rearrange(xt, 'b l d -> (b l) d'))

        query = rearrange(
            self.Q(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        key = rearrange(
            self.K(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        value = rearrange(
            self.V(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        scaled_dot_product = F.scaled_dot_product_attention(query, key, value)

        # print(scaled_dot_product.shape)
        out = rearrange(
            self.decoder(
                rearrange(scaled_dot_product, 'b l d -> (b l) d')
            ),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        return out

    @torch.no_grad()
    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size)
        """
        raise NotImplementedError("Equivariant divergence not implemented yet")

    def instantiate(self):
            return EquivariantTransformerFlowLJ(
                self.input_dim,
                self.embed_dim,
                self.key_dim,
                self.nbparticles
            )

class DirectionalTransformerFlowLJ(FlowNet):
    def __init__(self, input_dim, embed_dim, key_dim,  nbparticles):
        super().__init__()
        self.input_dim = input_dim
        self.nbparticles = nbparticles
        self.embed_dim = embed_dim
        self.key_dim = key_dim

        # we have multipe encoders to avoid the symmetry issue.
        self.encoder =  nn.Linear(self.input_dim + 1, self.embed_dim)
        self.decoder = nn.Linear(self.embed_dim, input_dim) # I guess this one can be shared for now and kept as a linear layer --> Note that if we have some MLP that's another call to jacobian
        self.Q = nn.Linear(self.embed_dim, self.key_dim)
        self.K = nn.Linear(self.embed_dim, self.key_dim)
        self.V = nn.Linear(self.input_dim+1, self.embed_dim) # we will modify this one to take this information of pairwise directionnality into account, as well as time

    def forward(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> This should scale to nbparticles in nd
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size, nbpart, dim)
        """


        repeat_t = repeat(t, 'b 1 -> b p 1', p=self.nbparticles)
        xt = torch.cat((x.clone(), repeat_t), dim=2)


        # reshape
        xt = rearrange(xt, 'b p d  -> p b d')
        xt_encoded = self.encoder(rearrange(xt, 'b l d -> (b l) d'))

        query = rearrange(
            self.Q(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )


        key = rearrange(
            self.K(xt_encoded),
            '(b l) d -> b l d',
            l = self.nbparticles
        )

        # compute the dot products
        # decompose the dotattention operation because we will need these matrices after
        r_q = repeat(query, 'b p d -> b p r d', r=self.nbparticles)
        r_k = repeat(key, 'b p d -> b r p d', r=self.nbparticles)
        # r_v = repeat(value, 'b p d -> b r p d', r=self.nbparticles)
        scale = 1 / r_q.shape[-1] ** 0.5
        dot_prod = (r_q * r_k).sum(-1) * scale # this is litterally the attention matrix
        sm = F.softmax(dot_prod, dim=-1) # (b, p, p)


        # compute the pairwise directional information
        x_r = repeat(x, 'b p d -> b p r1 d', r1=self.nbparticles)
        pairwise_directions = x_r - rearrange(x_r, 'b p r1 d -> b r1 p d') # (b, p, p, d)


        pairwise_directions_encoded = rearrange(
            torch.cat((pairwise_directions,repeat(t, 'b 1 -> b p p2 1', p=self.nbparticles, p2=self.nbparticles)), dim=-1), # (b, p, p, d+1),
            'b p p2 d -> (b p p2) d'
            )
        # compute the pairwise directional information
        values_ = self.V(pairwise_directions_encoded)

        value = rearrange(
            values_,
            '(b p1 p2) d -> b p1 p2 d',
            b = x.shape[0],
            p1 = self.nbparticles,
            p2 = self.nbparticles,
        )


        # compute the "scaled dot product" attention
        out = rearrange(
                self.decoder(
                    rearrange(
                        reduce(
                            value * sm.unsqueeze(-1),
                            'b p p2 d -> b p d',
                            'sum'
                        ),
                        'b p d -> (b p) d'
                    )
                ),
                '(b p) d -> b p d',
                p = self.nbparticles
            )

        return out

    @torch.no_grad()
    def divergence(self, x, t):
        """
        Inputs
        x :  (batch_size, nbpart, dim) --> weird but we want to try it in 2d first
        t :  (batch_size, 1)

        Outputs
        v :  (batch_size)
        """
        raise NotImplementedError("Equivariant divergence not implemented yet")

    def instantiate(self):
            return DirectionalTransformerFlowLJ(
                self.input_dim,
                self.embed_dim,
                self.key_dim,
                self.nbparticles
            )
