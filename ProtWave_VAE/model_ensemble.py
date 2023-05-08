
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


# =====================================================
# = Complete Un-supervised ProtWave-VAE model: p(x,z) =
# =====================================================


class ProtWaveVAE(nn.Module):
    """
    class description: This is the InfoVAE model. 
    """
    def __init__(
            self,
            DEVICE: str,
            encoder: any,
            decoder_recon: any,
            cond_mapper: any,
            z_dim: int = 6
        ):
        super(ProtWaveVAE, self).__init__()

        # model components:
        self.inference = encoder
        self.generator = decoder_recon
        self.cond_mapper = cond_mapper
        self.DEVICE = DEVICE

        # additional components:
        self.softmax = nn.Softmax(dim = -1)
        
        # hyperparameters
        self.z_dim = z_dim
	
    def reparam_trick(
            self,
            mu: torch.FloatTensor,
            var: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        function description: Samples z from a multivariate Gaussian with diagonal covariance matrix
        using the reparameterization trick.
        """
        std = var.sqrt()
        eps = torch.rand_like(std)
        z = mu + eps * std # latent code
        return z
	
    def forward(self, x: torch.FloatTensor) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ):
        """
        - data shapes -
        data:
                x --> (batch_size, protein_len, aa_labels)
                mu,var --> (batch_size, z_dim), (batch_size, z_dim)
		z --> (batch_size, z_dim)
		z_upscale --> (batch_size, 1, protein_len)
		logits_xrc --> (batch_size, protein_len, aa_labels)
		y_pred --> (batch_size, 1)
        """
        # q(mu, var|x)
        z_mu, z_var = self.inference(x.permute(0, 2, 1))
        # q(z|x)
        z = self.reparam_trick(z_mu, z_var)
        # upscale latent code
        z_upscale = self.cond_mapper(z)
        # p(x|z)
        logits_xrc = self.generator(x.permute(0,2, 1), z_upscale).permute(0,2,1)

        return (
                logits_xrc,
                z, 
                z_mu,
                z_var
        )

    @staticmethod
    def compute_kernel(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:

        # size of the mini batches
        x_size, y_size = x.shape[0], y.shape[0]

        # dimension based on z size
        dim = x.shape[1] # can also be considered as a hyperparameter

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        x_core = x.expand(x_size, y_size, dim)
        y_core = y.expand(x_size, y_size, dim)

        return torch.exp(-(x_core-y_core).pow(2).mean(2)/dim)

    @staticmethod
    def compute_mmd(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        function description: compute the max-mean discrepancy
        arg:
            x --> random distribution z~p(x)
            y --> embedding distribution z'~q(z)
        return:
            MMD_loss --> max-mean discrepancy loss between the sampled noise
                  and embedded distribution
        """

        x_kernel = SS_InfoVAE.compute_kernel(x,x)
        y_kernel = SS_InfoVAE.compute_kernel(y,y)
        xy_kernel = SS_InfoVAE.compute_kernel(x,y)
        return x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
      
    def compute_loss(
            self,
            xr: torch.FloatTensor,
            x: torch.FloatTensor,
            z_pred: torch.FloatTensor,
            true_samples: torch.FloatTensor,
            z_mu: torch.FloatTensor,
            z_var: torch.FloatTensor
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor
        ):

        # POSTERIOR KL-DIVERGENCE loss:
        loss_kld = torch.mean(-0.5 * torch.sum(1 + z_var.log() - z_mu ** 2 - z_var, dim = 1), dim = 0)
        # MMD loss: 
        loss_mmd = SS_InfoVAE.compute_mmd(true_samples, z_pred) # mmd (reg.) loss
        # RECONSTRUCTION loss:
        nll = nn.CrossEntropyLoss(reduction = 'none') # reconstruction loss
        x_nums = torch.argmax(x, dim = -1).long() # convert ground truth from one hot to num. rep.
        loss_nll = nll(xr.permute(0, 2, 1), x_nums) # nll for reconstruction
        #loss_nll = torch.sum(loss_nll, dim = -1) # sum nll along protein sequence
        loss_nll = torch.mean(loss_nll, dim = -1) # average nll along protein sequence
        loss_nll = torch.mean(loss_nll) # average over the batch
                   
 
        return (
                loss_nll,
                loss_kld,
                loss_mmd,
        )
    @torch.no_grad()
    def aa_sample(
            self,
            X: torch.FloatTensor,
            option: str='categorical'
        ) -> torch.FloatTensor:
        onehot_transformer = torch.eye(21)

        if option=='categorical': # sample from a categorical distribution
            cate = torch.distributions.Categorical(X)
            X = cate.sample()
        
        else: # sample from an argmax distribution
            X = torch.argmax(X, dim = -1)

        return onehot_transformer[X]

    @torch.no_grad()
    def sample(
            self,
            args: any,
            X_context: torch.FloatTensor,
            z: torch.FloatTensor,
            option: str='categorical',
        ) -> torch.FloatTensor:

        # eval model (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate

	# init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                               protein_len+1,
                                                               1,
                                                               1
        ).to(args.DEVICE) # [B, L+1, L, 21]

        # upscale latent code
        z_context = self.cond_mapper(z) # linear transformation: [B,6] -> [B, L, 21]
        # generate first index (only latent code conditioning)
        X_gen_logits = self.generator(
                                    X_context[:,0,:,:].permute(0,2,1),
                                    z_context
        ).permute(0, 2, 1) # [B, L, 21]
        # insert amino acid label in the first position
        X_temp[:,0,:] = self.aa_sample(X_gen_logits.softmax(dim=-1), option=option)[:,0]
        # first index of the context is the probability prediction with only latent conditional
        X_context[:,0,:,:] = X_gen_logits.softmax(dim = -1)

        for ii in tqdm(range(1, protein_len)):

            # make logit predictions for the remaining positions
            X_gen_logits = self.generator(
                                    X_temp[:,:,:].permute(0,2,1),
                                    z_context
            ).permute(0,2,1)
            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_gen_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_gen_logits.softmax(dim=-1)
            # update the
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]

        # last index is the final latent-based AR prediction
        X_context[:,-1,:,:] = X_temp
        return X_context


    @torch.no_grad()
    def diversify(
        self,
        args: any,
        X_context: torch.FloatTensor,
        z: torch.FloatTensor,
        L: int=1,
        option: str='categorical'
        ) -> torch.FloatTensor:
       
        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()
        
        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate
        
	# init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                                   protein_len+1,
                                                                   1,
                                                                   1
        ).to(args.DEVICE) # [B, L+1, L, 21]
        
        # insert the conditioned amino acids
        X_temp[:,:L,:] = X_template[:,:L,:]
        X_context[:,:,:L,:] = X_template.unsqueeze(1).repeat(
                                                        1,
                                                        protein_len+1,
                                                        1,
                                                        1
        )[:,:,:L,:]


        # upscale latent code
        z_context = self.cond_mapper(z) # linear transformation: [B,6] -> [B, L, 21]
       
        for ii in tqdm(range(L, protein_len)):
            
            # make logit predictions for the remaining positions 
            X_gen_logits = self.generator(
                                    X_temp[:,:,:].permute(0,2,1),
                                    z_context
            ).permute(0,2,1)
            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_gen_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_gen_logits.softmax(dim=-1)
            # update the 
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]
         
        # last index is the final latent-based AR prediction
        X_context[:,-1,:,:] = X_temp
        
        return X_context


    def create_uniform_tensor(
            self,
            args: any,
            X: torch.FloatTensor,
            option: str='random'
        ) -> torch.FloatTensor:

        batch_size, seq_length, aa_length = X.shape

        X_temp = torch.ones_like(X)

        if option == 'random':
            

            X_temp[:,:,-1] = X_temp[:,:,-1]*0 # give no prob to the padded tokens
            X_temp[:,:,:-1] = X_temp[:,:,:-1] / (aa_length-1)

        elif option == 'guided':

            X[:,:,-1] = X[:,:,-1]*0 # remove padded tokens
            X_temp = X_temp - X # removev the ground truth labels
            X_temp[:,:,-1] = X_temp[:,:,-1]*0 # give no prob to the padded tokens
            X_temp[:,:,:-1] = X_temp[:,:,:-1] / (aa_length-2)

        return X_temp

    @torch.no_grad()
    def randomly_diversify(
        self,
        args: any,
        X_context: torch.FloatTensor,
        L: int=1,
        option: str='categorical'
        ) -> torch.FloatTensor:


        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate

        # init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                                       protein_len+1,
                                                                       1,
                                                                       1
        ).to(args.DEVICE) # [B, L+1, L, 21]

        # insert the conditioned amino acids
        X_temp[:,:L,:] = X_template[:,:L,:]
        X_context[:,:,:L,:] = X_template.unsqueeze(1).repeat(
                                                            1,
                                                            protein_len+1,
                                                            1,
                                                            1
        )[:,:,:L,:]


        for ii in tqdm(range(L, protein_len)):


            # make logit predictions for the remaining positions
            X_logits = self.create_uniform_tensor(
                        args=args,
                        X=X_template
            )

            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_logits.softmax(dim=-1)
            # update the context
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]

            # last index is the final latent-based AR prediction
            X_context[:,-1,:,:] = X_temp

        return X_context

    def pick_pos2mut(self, list_pos: list) -> (
            list,
            int
        ):

        position = np.random.choice((list_pos))
        list_pos.remove(position)

        return (
                list_pos,
                position
        )

    @torch.no_grad()
    def guided_randomly_diversify(
            self,
            args: any,
            X_context: torch.FloatTensor,
            X_design: torch.FloatTensor,
            L: int=1,
            min_leven_dists: list=[],
            option: str='categorical',
            design_seq_lens: list=[],
            ref_seq_len: int=100,
            num_gaps: int=0
        ) -> torch.FloatTensor:

        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the max seq
        n = X_context.shape[0] # number of sequences to generate

        # init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
      
        # insert the whole instead of only the conditional info
        X_temp[:,:,:] = X_template[:,:,:]
        X_context = X_template.unsqueeze(1).repeat(
                1,
                protein_len+1,
                1,
                1
        )[:,:,:,:]
        

        # number of sites that fit along the length of the reference sequence
        ref_window_size = (ref_seq_len - L)

        for ii, min_leven_dist in enumerate(min_leven_dists): # how many times to mutate positions
            
            # positions that are allowed to be mutated
            list_pos = [ii for ii in range(L, ref_seq_len)] # get mutating positions
            
            diff = 0 # no need to replace gaps with amino acids
           
            if int(min_leven_dist) > int(ref_window_size):
                
                diff = int(min_leven_dist) - len(list_pos)
                # create new list position to account for longer sequence
                list_pos = [ii for ii in range(L, ref_seq_len + diff)]

            for jj in range(int(min_leven_dist)):

                list_pos, pos_idx = self.pick_pos2mut(list_pos=list_pos)
                # make logit predictions for the remaining positions
                X_logits = self.create_uniform_tensor(
                                    args=args,
                                    X=X_template,
                                    option='guided'
                )

                # insert amino acid at the next position
                X_temp[ii,pos_idx,:] = self.aa_sample(X_logits)[ii,pos_idx]
                # update the next index of the conditional tensor
                X_context[ii,jj,pos_idx,:] = X_logits[ii,pos_idx]
                # last index is the final sample
                X_context[ii,-1,pos_idx,:] = X_temp[ii,pos_idx,:]
          
            # fill in gaps
            X_context[ii,-1,-(num_gaps-diff):,:-1] = 0
            X_context[ii,-1,-(num_gaps-diff):, -1] = 1
                

        print(f'Length start {L} and list positions:', list_pos)
        return X_context
       

# =========================================================
# = Complete Semi-supervised ProtWave-VAE model: p(x,y,z) =
# =========================================================



class SS_ProtWaveVAE(nn.Module):
    """
    class description: This is the InfoVAE model.
    """
    def __init__(
            self,
            DEVICE: str,
            encoder: any,
            decoder_recon: any,
            cond_mapper: any,
            decoder_pheno: any,
            z_dim: int = 6
        ):
        super(SS_ProtWaveVAE, self).__init__()

        # model components:
        self.inference = encoder
        self.generator = decoder_recon
        self.cond_mapper = cond_mapper
        self.discriminator = decoder_pheno
        self.DEVICE = DEVICE

        # additional components:
        self.softmax = nn.Softmax(dim = -1)

        # hyperparameters
        self.z_dim = z_dim

    def reparam_trick(
            self,
            mu: torch.FloatTensor,
            var: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        function description: Samples z from a multivariate Gaussian with diagonal covariance matrix
        using the reparameterization trick.
        """
        std = var.sqrt()
        eps = torch.rand_like(std)
        z = mu + eps * std # latent code
        return z

    def forward(self, x: torch.FloatTensor) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        - data shapes -
        data:
                x --> (batch_size, protein_len, aa_labels)
                mu,var --> (batch_size, z_dim), (batch_size, z_dim)
		z --> (batch_size, z_dim)
		z_upscale --> (batch_size, 1, protein_len)
		logits_xrc --> (batch_size, protein_len, aa_labels)
		y_pred --> (batch_size, 1)
        """
        # q(mu, var|x)
        z_mu, z_var = self.inference(x.permute(0, 2, 1))
        # q(z|x)
        z = self.reparam_trick(z_mu, z_var)
        # upscale latent code
        z_upscale = self.cond_mapper(z)
        # p(x|z)
        logits_xrc = self.generator(x.permute(0,2, 1), z_upscale).permute(0,2,1)
        # p(y|z)
        y_pred_R, y_pred_C = self.discriminator(z)

        return (
                logits_xrc,
                y_pred_R,
                y_pred_C,
                z,
                z_mu,
                z_var
        )

    @staticmethod
    def compute_kernel(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:

        # size of the mini batches
        x_size, y_size = x.shape[0], y.shape[0]

        # dimension based on z size
        dim = x.shape[1] # can also be considered as a hyperparameter

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        x_core = x.expand(x_size, y_size, dim)
        y_core = y.expand(x_size, y_size, dim)

        return torch.exp(-(x_core-y_core).pow(2).mean(2)/dim)

    @staticmethod
    def compute_mmd(
            x: torch.FloatTensor,
            y: torch.FloatTensor
        ) -> torch.FloatTensor:
        """
        function description: compute the max-mean discrepancy
        arg:
            x --> random distribution z~p(x)
            y --> embedding distribution z'~q(z)
        return:
            MMD_loss --> max-mean discrepancy loss between the sampled noise
                  and embedded distribution
        """

        x_kernel = SS_InfoVAE.compute_kernel(x,x)
        y_kernel = SS_InfoVAE.compute_kernel(y,y)
        xy_kernel = SS_InfoVAE.compute_kernel(x,y)
        return x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()

    def compute_loss(
            self,
            xr: torch.FloatTensor,
            x: torch.FloatTensor,
            y_pred_R: torch.FloatTensor,
            y_true_R: torch.FloatTensor,
            y_pred_C: torch.FloatTensor,
            y_true_C: torch.FloatTensor,
            z_pred: torch.FloatTensor,
            true_samples: torch.FloatTensor,
            z_mu: torch.FloatTensor,
            z_var: torch.FloatTensor
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor
        ):

        # POSTERIOR KL-DIVERGENCE loss:
        loss_kld = torch.mean(-0.5 * torch.sum(1 + z_var.log() - z_mu ** 2 - z_var, dim = 1), dim = 0)
        # MMD loss:
        loss_mmd = SS_InfoVAE.compute_mmd(true_samples, z_pred) # mmd (reg.) loss
        # RECONSTRUCTION loss:
        nll = nn.CrossEntropyLoss(reduction = 'none') # reconstruction loss
        x_nums = torch.argmax(x, dim = -1).long() # convert ground truth from one hot to num. rep.
        loss_nll = nll(xr.permute(0, 2, 1), x_nums) # nll for reconstruction
        #loss_nll = torch.sum(loss_nll, dim = -1) # sum nll along protein sequence
        loss_nll = torch.mean(loss_nll, dim = -1) # average nll along protein sequence
        loss_nll = torch.mean(loss_nll) # average over the batch
        # DISCRIMINATION loss:

        try:
            # for classification loss
            loss_pheno_BCE = nn.BCELoss()
            loss_pheno_C = loss_pheno_BCE(y_pred_C, y_true_C)

            # for regression loss
            loss_pheno_MSE = nn.MSELoss()
            loss_pheno_R = loss_pheno_MSE(y_pred_R, y_true_R)

            loss_pheno = loss_pheno_R + loss_pheno_C

        except RuntimeErrors: # if the whole batch didn't have experimental true labels
            loss_pheno = torch.tensor([0]).to(self.DEVICE)


        return (
                loss_nll,
                loss_kld,
                loss_mmd,
                loss_pheno
        )
    @torch.no_grad()
    def aa_sample(
            self,
            X: torch.FloatTensor,
            option: str='categorical'
        ) -> torch.FloatTensor:
        onehot_transformer = torch.eye(21)

        if option=='categorical': # sample from a categorical distribution
            cate = torch.distributions.Categorical(X)
            X = cate.sample()

        else: # sample from an argmax distribution
            X = torch.argmax(X, dim = -1)

        return onehot_transformer[X]

    @torch.no_grad()
    def sample(
            self,
            args: any,
            X_context: torch.FloatTensor,
            z: torch.FloatTensor,
            option: str='categorical',
        ) -> torch.FloatTensor:

        # eval model (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate

	# init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                               protein_len+1,
                                                               1,
                                                               1
        ).to(args.DEVICE) # [B, L+1, L, 21]

        # upscale latent code
        z_context = self.cond_mapper(z) # linear transformation: [B,6] -> [B, L, 21]
        # generate first index (only latent code conditioning)
        X_gen_logits = self.generator(
                                    X_context[:,0,:,:].permute(0,2,1),
                                    z_context
        ).permute(0, 2, 1) # [B, L, 21]
        # insert amino acid label in the first position
        X_temp[:,0,:] = self.aa_sample(X_gen_logits.softmax(dim=-1), option=option)[:,0]
        # first index of the context is the probability prediction with only latent conditional
        X_context[:,0,:,:] = X_gen_logits.softmax(dim = -1)

        for ii in tqdm(range(1, protein_len)):

            # make logit predictions for the remaining positions
            X_gen_logits = self.generator(
                                    X_temp[:,:,:].permute(0,2,1),
                                    z_context
            ).permute(0,2,1)
            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_gen_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_gen_logits.softmax(dim=-1)
            # update the
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]

        # last index is the final latent-based AR prediction
        X_context[:,-1,:,:] = X_temp
        return X_context


    @torch.no_grad()
    def diversify(
        self,
        args: any,
        X_context: torch.FloatTensor,
        z: torch.FloatTensor,
        L: int=1,
        option: str='categorical'
        ) -> torch.FloatTensor:

        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate

	# init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                                   protein_len+1,
                                                                   1,
                                                                   1
        ).to(args.DEVICE) # [B, L+1, L, 21]

        # insert the conditioned amino acids
        X_temp[:,:L,:] = X_template[:,:L,:]
        X_context[:,:,:L,:] = X_template.unsqueeze(1).repeat(
                                                        1,
                                                        protein_len+1,
                                                        1,
                                                        1
        )[:,:,:L,:]


        # upscale latent code
        z_context = self.cond_mapper(z) # linear transformation: [B,6] -> [B, L, 21]

        for ii in tqdm(range(L, protein_len)):

            # make logit predictions for the remaining positions
            X_gen_logits = self.generator(
                                    X_temp[:,:,:].permute(0,2,1),
                                    z_context
            ).permute(0,2,1)
            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_gen_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_gen_logits.softmax(dim=-1)
            # update the
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]

        # last index is the final latent-based AR prediction
        X_context[:,-1,:,:] = X_temp

        return X_context



    def create_uniform_tensor(
            self,
            args: any,
            X: torch.FloatTensor,
            option: str='random'
        ) -> torch.FloatTensor:

        batch_size, seq_length, aa_length = X.shape

        X_temp = torch.ones_like(X)

        if option == 'random':


            X_temp[:,:,-1] = X_temp[:,:,-1]*0 # give no prob to the padded tokens
            X_temp[:,:,:-1] = X_temp[:,:,:-1] / (aa_length-1)

        elif option == 'guided':

            X[:,:,-1] = X[:,:,-1]*0 # remove padded tokens
            X_temp = X_temp - X # removev the ground truth labels
            X_temp[:,:,-1] = X_temp[:,:,-1]*0 # give no prob to the padded tokens
            X_temp[:,:,:-1] = X_temp[:,:,:-1] / (aa_length-2)

        return X_temp

    @torch.no_grad()
    def randomly_diversify(
        self,
        args: any,
        X_context: torch.FloatTensor,
        L: int=1,
        option: str='categorical'
        ) -> torch.FloatTensor:


        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the maximum sequence
        n = X_context.shape[0] # number of sequences to generate

        # init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]
        X_context = torch.zeros_like(X_context).unsqueeze(1).repeat(1,
                                                                       protein_len+1,
                                                                       1,
                                                                       1
        ).to(args.DEVICE) # [B, L+1, L, 21]

        # insert the conditioned amino acids
        X_temp[:,:L,:] = X_template[:,:L,:]
        X_context[:,:,:L,:] = X_template.unsqueeze(1).repeat(
                                                            1,
                                                            protein_len+1,
                                                            1,
                                                            1
        )[:,:,:L,:]


        for ii in tqdm(range(L, protein_len)):


            # make logit predictions for the remaining positions
            X_logits = self.create_uniform_tensor(
                        args=args,
                        X=X_template
            )

            # insert amino acid at the next position
            X_temp[:,ii,:] = self.aa_sample(X_logits.softmax(dim=-1))[:,ii]
            # update the next index of the conditional tensor
            X_context[:,ii,:,:] = X_logits.softmax(dim=-1)
            # update the context
            X_context[:,ii,:ii,:] = X_temp[:,:ii,:]

            # last index is the final latent-based AR prediction
            X_context[:,-1,:,:] = X_temp

        return X_context


    def pick_pos2mut(self, list_pos: list) -> (
            list,
            int
        ):

        position = np.random.choice((list_pos))
        list_pos.remove(position)

        return (
                list_pos,
                position
        )

    @torch.no_grad()
    def guided_randomly_diversify(
            self,
            args: any,
            X_context: torch.FloatTensor,
            X_design: torch.FloatTensor,
            L: int=1,
            min_leven_dists: list=[],
            option: str='categorical',
            design_seq_lens: list=[],
            ref_seq_len: int=100,
            num_gaps: int=0
        ) -> torch.FloatTensor:

        # copy context sequence to track the conditioned amino acids
        X_template = X_context.clone()

        # eval mode (important, especially with BatchNorms)
        self.eval()

        # misc helper variables/objects
        protein_len = X_context.shape[1] # length of the max seq
        n = X_context.shape[0] # number of sequences to generate

        # init. placeholder tensors
        X_temp = torch.zeros_like(X_context).to(args.DEVICE) # [B, L, 21]

        # insert the whole instead of only the conditional info
        X_temp[:,:,:] = X_template[:,:,:]
        X_context = X_template.unsqueeze(1).repeat(
                1,
                protein_len+1,
                1,
                1
        )[:,:,:,:]


        # number of sites that fit along the length of the reference sequence
        ref_window_size = (ref_seq_len - L)

        for ii, min_leven_dist in enumerate(min_leven_dists): # how many times to mutate positions

            # positions that are allowed to be mutated
            list_pos = [ii for ii in range(L, ref_seq_len)] # get mutating positions

            diff = 0 # no need to replace gaps with amino acids

            if int(min_leven_dist) > int(ref_window_size):

                diff = int(min_leven_dist) - len(list_pos)
                # create new list position to account for longer sequence
                list_pos = [ii for ii in range(L, ref_seq_len + diff)]

            for jj in range(int(min_leven_dist)):

                list_pos, pos_idx = self.pick_pos2mut(list_pos=list_pos)
                # make logit predictions for the remaining positions
                X_logits = self.create_uniform_tensor(
                                    args=args,
                                    X=X_template,
                                    option='guided'
                )

                # insert amino acid at the next position
                X_temp[ii,pos_idx,:] = self.aa_sample(X_logits)[ii,pos_idx]
                # update the next index of the conditional tensor
                X_context[ii,jj,pos_idx,:] = X_logits[ii,pos_idx]
                # last index is the final sample
                X_context[ii,-1,pos_idx,:] = X_temp[ii,pos_idx,:]

            # fill in gaps
            X_context[ii,-1,-(num_gaps-diff):,:-1] = 0
            X_context[ii,-1,-(num_gaps-diff):, -1] = 1


        print(f'Length start {L} and list positions:', list_pos)
        return X_context
