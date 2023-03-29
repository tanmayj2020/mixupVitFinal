import torch 
import torch.nn as nn




class MLP(nn.Module):
    """
    NN before attention block 
    Parameters 
    ----------
    input_dim : int 
        Number of input features
    output_dim : int 
        Number of output features 
    
    hidden_dim : int 
        Number of hidden dimensions
    
    p : float 
        Dropout probability 
    Attribute
    ---------
    fc : nn.Linear 
        First linear layer
    act : nn.GELU 
        GELU activation function 
    fc2 : nn.Linear 
        The second linear layer
    
    drop : nn.Dropout 
        Dropout Layer
    """

    def __init__(self , input_dim , hidden_dim , output_dim , p):
        super().__init__()
        self.fc1 = nn.Linear(input_dim , hidden_dim )
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim , output_dim)
        self.drop = nn.Dropout(p)

    def forward(self , x):
        """
        Forward of multi layer perceptron
        Parameters 
        ----------
        x: torch.Tensor
            Shape = (n_samples , n_patches + 1 ,input_dim)
        Returns 
        -------
        torch.Tensor 
            Shape = (n_samples , n_patches + 1 ,output_dim)       
        """
        x = self.fc1(x) #(n_samples , n_patches + 1,hidden_dim)
        x = self.act1(x) #(n_samples , n_patches + 1, hidden_dim)
        x = self.fc2(x) # (n_samples , n_patches + 1 , output_dim)
        x = self.drop(x) #(n_samples , n_patches + 1, output_dim)
        return x



class Attention(nn.Module):
    """
    Implements attention 
    Parameters 
    ----------
    dim :int 
        Input dimension of the per token feature
    
    n_head : int 
        Number of attention heads
    
    qkv_bias : bool 
        If True then we include bias to query , key and value projections
    
    atten_p : float 
        Dropout probability of attention matrix 
    
    proj_p : float 
        Dropout probability of the projecttion to output tensor 
    Attributes
    ----------
    scale : float 
        Prevents softmax output low gradients
    qkv : 
        Linear projection of input embedding to query , key and value 
    proj :
        Linear mapping from concateated output of attention to same space
    attn_drop , proj_drop : nn.Dropout 
        Applying dropouts to prevent overfitting 
    """

    def __init__(self , input_dim , n_head = 12 , qkv_bias = True , attn_prob = 0. , proj_prob = 0.):
        super().__init__()
        self.input_dim = input_dim 
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(input_dim , input_dim * 3 , bias = qkv_bias)
        self.attn_drop = nn.Dropout(p = attn_prob)
        self.proj = nn.Linear(input_dim , input_dim)
        self.proj_drop = nn.Dropout(p = proj_prob)
    

    def forward(self , x , perm_x = None , lmbda= None , mixup_block = False, mixup_type=None):
        """
        Forward pass attention block 
        Parameters
        ----------
        x : torch.Tensor
            Shape = (n_samples,n_patches + 1 , input_dim)
        
        Returns 
        -------
        torch.Tensor
            Shape = (n_samples , n_patches + 1, input_dim)
        
        
        """
        if mixup_block == False:
            n_samples , n_tokens , input_dim = x.shape
            qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
            qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
            qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
            q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
            k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
            dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
            attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
            attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
            weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
            weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
            weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
            x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
            output = self.proj_drop(x)
            return output
        else:
            assert perm_x is not None
            assert lmbda is not None
            if mixup_type == "attention":
                
                def pre_attention(x):
                    n_samples , n_tokens , input_dim = x.shape
                    qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                    qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                    qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                    q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                    k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
                    dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                    attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                    attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
                    return attn
                
                n_samples , n_tokens , input_dim = x.shape
                qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
                dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
                
                #NOTE : ATTENTION MIXUP STEP======================================
                attn_perm = pre_attention(perm_x)
                attn = lmbda * attn + (1 - lmbda) * attn_perm                                
                #==================================================================
                weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
                weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
                weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
                x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
                output = self.proj_drop(x)
                return output
                    
            if mixup_type == "key_and_value":
                def pre_kv(x):
                    n_samples , n_tokens , input_dim = x.shape
                    qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                    qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                    qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                    q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                    return q ,k ,v

                n_samples , n_tokens , input_dim = x.shape
                qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                
                # NOTE : Applying mixup here
                q_perm , k_perm , v_perm = pre_kv(perm_x)
                k = lmbda * k + (1 - lmbda) * k_perm
                v = lmbda * v + (1 - lmbda) * v_perm

                k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
                dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
                weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
                weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
                weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
                x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
                output = self.proj_drop(x)
                return output
                
                
            elif mixup_type == "query":
                def pre_q(x):
                    n_samples , n_tokens , input_dim = x.shape
                    qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                    qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                    qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                    q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                    return q ,k ,v

                n_samples , n_tokens , input_dim = x.shape
                qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
                qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
                qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
                q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
                
                # NOTE : Applying mixup here
                q_perm , k_perm , v_perm = pre_q(perm_x)
                q = lmbda * q + (1 - lmbda) * q_perm
                
                k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
                dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
                attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
                weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
                weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
                weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
                x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
                output = self.proj_drop(x)
                return output



class PatchEmbed(nn.Module):
    """
    Splits the image into patches and embed the patches 
    
    Parameters 
    ----------
    img_size : int 
        Size of input image(W == H)
    
    patch_size : int 
        Size of the patch(W == H)
    
    embed_dimension : int 
        Size of patch embedding   
    
    Attributes
    -----------
    n_patches : int 
        Number of patches of the image
    
    proj : nn.Conv2d
        Convolution to convert does both splitting into patches and embedding the patches
    """


    def __init__(self , img_size , patch_size ,in_channels = 3 , embed_dimensions = 768 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels , embed_dimensions , kernel_size=patch_size , stride=patch_size)

    def forward(self , x):
        """
        Forward pass of the Module 
        Parameters
        ----------
        x : torch.Tensor
            Shape - (n_samples , input_channels , img_size , img_size)
        
        Output 
        -------
        torch.Tensor
            Shape - (n_samples , n_patches , embed_dimensions)        
        
        """
        x = self.proj(x) # (n_samples , embed_dim , n_patch ** 0.5 , n_patch ** 0.5)
        x = x.flatten(2) # (n_samples , embed_dim , n_patch)
        x = x.transpose(1 ,2 ) #(n_samples , n_patches , embed_dim)
        return x

    



class Block(nn.Module):
    """
    Transformer block 
    Parameters
    ----------
    dim : int 
        Input token dimension 
    
    n_head : int 
        Number of attention heads
    
    mlp_ration : float 
        Determines the number of hidden units in MLP module with respect to dim 
    
    qkv_bias : bool 
        If true we include bias to quuery , key and value 
    p , attn_prob : float
        Dropout probability
    Attributes
    ----------
    norm1 , norm2 : nn.LayerNorm
    attn : Attention Module
    mlp : MLP modules
    """
    def __init__(self , dim , n_head , mlp_ratio =4.0 , qkv_bias = True ,p = 0. , attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim ,eps = 1e-6)
        self.attn = Attention(dim , n_head , qkv_bias, attn_p , p)
        self.norm2 = nn.LayerNorm(dim , eps = 1e-6)
        self.mlp = MLP(dim , int(dim * mlp_ratio) , dim  , p)

    def forward(self , x , perm_x = None , lmbda= None , mixup_block = False, mixup_type=None):
        """
        Forward pass of transformer block 
        Parameters
        ----------
        x : torch.Tensor
            Shape =  (n_samples , n_patches + 1 , dim)
        
        Returns 
        -------
        torch.Tensor
            Shape = (n_samples, n_patches +1 , dim)
        """
        if mixup_block:
            assert perm_x is not None
            assert lmbda is not None
            x = x + self.attn(self.norm1(x) , self.norm1(perm_x) , lmbda , True , mixup_type)
        else:           
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """
    Vision Transformer Class
    Parameters
    ----------
    img_size : int
        Both height and width of the image(square)
    patch_size : int 
        Both height and width of the patch (sqaure)
    in_channels : 
        Number of input channels   
    n_classes : int
        Number of classes
    mlp_ratio : float 
        Hidden layer ration of the MLP module 
    qkv_bias : bool     
        If true then we include bias to query , key and value matrix
    p , attn_p : float 
        Dropout Probability 
        
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of PatchEmbed layer
    
    cls_token : nn.Parameter
        Learnable parameter that represents the first token in the sequence 
        "It has dimensions of embed dimensions"
    pos_emb : nn.Parameter
        Positional embedding of cls token + all the the patches
        It has (n_patches +1 ) * embed_dim elements
    pos_drop : nn.Dropout 
        Dropout Layer
    blocks : nn.ModuleList 
        List of 'Block' module
    norm : nn.LayerNorm 
        Layer Normalisation 
    """
    def __init__(self , img_size = 384 , patch_size = 16 , in_chans = 3 , n_classes = 1000 , embed_dim = 768 , depth = 12 , n_heads = 12 , mlp_ratio = 4. , qkv_bias = True , p = 0.1 , attn_p = 0.1):
        super().__init__()
        self.patch_embed =  PatchEmbed(img_size , patch_size , in_chans , embed_dim )
        self.cls_token = nn.Parameter(torch.zeros(1 ,1 , embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1 , 1 + self.patch_embed.n_patches , embed_dim))
        self.pos_drop = nn.Dropout(p = p)
        self.blocks  = nn.ModuleList(
            [Block(embed_dim , n_heads , mlp_ratio , qkv_bias , p , attn_p) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim , eps =1e-6)
        self.head = nn.Linear(embed_dim , n_classes)
    

    def forward(self , x , perm_x=None , mixup_type = None , lmbda=None, block_num =None):
        """
        Run the forward pass
        Parameters 
        ----------
        x : torch.Tensor
            Shape = (n_samples , in_channels , img_size , img_size)
        Returns 
        -------
        logits : torch.Tensor 
            Logits over all classes Shape = (n_samples , n_classes)            
        """
        def forward_before_block(x):
            n_samples = x.shape[0]
            x = self.patch_embed(x) #(n_samples , n_patches , embed_dim)
            cls_token = self.cls_token.expand(n_samples , - 1, -1) #(n_samples , 1 , embed_dim)
            x = torch.cat((cls_token , x) , dim = 1) #(n_samples , 1 + n_patches, embed_dim)
            x = x + self.pos_emb #(n_samples , n_patches + 1, embed_dim)
            x = self.pos_drop(x) #(n_samples , n_patches +1 , embed_dim)
            return x
        

        if mixup_type == "none" or mixup_type==None:
            x = forward_before_block(x)
            for blocks in self.blocks:
                x = blocks(x)            #(n_samples, n_patches + 1, embed_dim)
        
        else:   
            x = forward_before_block(x)
            perm_x = forward_before_block(perm_x)
            count = 0 
            for block in self.blocks:
                if count == block_num:
                    x = block(x , perm_x , lmbda , True, mixup_type)
                if count < block_num:
                    x = block(x)
                    perm_x = block(perm_x)
                if count > block_num:
                    x = block(x)
                count += 1 

        x = self.norm(x) #(n_samples, n_patches + 1, embed_dim)
        cls_token_final = x[: ,0] #Just the class tokens (n_samples ,1 , embed_dim)
        x = self.head(cls_token_final) #(n_samples, n_class)
        return x
