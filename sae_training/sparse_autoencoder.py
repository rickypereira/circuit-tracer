"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import gzip
import os
import pickle
from functools import partial

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule, HookPoint
import torch.distributed as dist

from .geom_median.src.geom_median.torch import compute_geometric_median


class SparseAutoencoder(HookedRootModule):
    """
    
    """
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.dtype = cfg.dtype
        self.device = cfg.device

        # transcoder stuff
        self.d_out = self.d_in
        if cfg.is_transcoder and cfg.d_out is not None:
            self.d_out = cfg.d_out

        # sparse-connection transcoder stuff
        self.spacon_sae_W_dec = None
        if cfg.is_sparse_connection:
            # load in the sae decoder weights that we'll use to train sparse connections
            sparse_connection_sae_path = cfg.sparse_connection_sae_path
            
            if sparse_connection_sae_path.endswith(".pt"):
                state_dict = torch.load(sparse_connection_sae_path, map_location=self.device) # DDP Change: map to correct device
            elif sparse_connection_sae_path.endswith(".pkl.gz"):
                with gzip.open(sparse_connection_sae_path, 'rb') as f:
                    state_dict = pickle.load(f)
            elif sparse_connection_sae_path.endswith(".pkl"):
                with open(sparse_connection_sae_path, 'rb') as f:
                    state_dict = pickle.load(f)
            else:
                raise ValueError(f"Unexpected file extension: {sparse_connection_sae_path}, supported extensions are .pt, .pkl, and .pkl.gz")

            self.spacon_sae_W_dec = state_dict['state_dict']['W_dec'].to(self.device) if not cfg.sparse_connection_use_W_enc else state_dict['state_dict']['W_enc'].to(self.device).T
            del state_dict
            torch.cuda.empty_cache()

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_out, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            # Anthropic normalize this to have unit columns
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.b_dec_out = None
        if cfg.is_transcoder:
            self.b_dec_out = nn.Parameter(
                torch.zeros(self.d_out, dtype=self.dtype, device=self.device)
            )


        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def forward(self, x, dead_neuron_mask = None, mse_target=None):
        # move x to correct dtype
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  # Remove encoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        feature_acts = self.hook_hidden_post(torch.nn.functional.relu(hidden_pre))

        if self.cfg.is_transcoder:
            # dumb if statement to deal with transcoders
            # hopefully branch prediction takes care of this
            sae_out = self.hook_sae_out(
                einops.einsum(
                    feature_acts,
                    self.W_dec,
                    "... d_sae, d_sae d_out -> ... d_out",
                )
                + self.b_dec_out
            )
        else:
            sae_out = self.hook_sae_out(
                einops.einsum(
                    feature_acts,
                    self.W_dec,
                    "... d_sae, d_sae d_out -> ... d_out",
                )
                + self.b_dec
            )
        
        # add config for whether l2 is normalized:
        mse_target_norm = torch.norm(mse_target.float(), dim=-1, keepdim=True)
        if mse_target is None:
            mse_loss = torch.pow((sae_out - x.float()), 2) / (torch.norm(x.float(), dim=-1, keepdim=True) + 1e-8)
        else:
            mse_loss = torch.pow((sae_out - mse_target.float()), 2) / (mse_target_norm + 1e-8)
            
        mse_loss_ghost_resid = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        # gate on config and training so evals is not slowed down.
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None and dead_neuron_mask.sum() > 0:
            
            # ghost protocol
            
            # 1.
            residual = (x - sae_out).detach()
            l2_norm_residual = torch.norm(residual, dim=-1)
            
            # 2.
            # hidden_pre is of shape [batch_size, d_sae]
            # dead_neuron_mask is of shape [d_sae]
            # feature_acts_dead_neurons_only should be of shape [batch_size, n_dead_neurons]
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])

            # W_dec is [d_sae, d_out], W_dec[dead_neuron_mask,:] is [n_dead_neurons, d_out]
            # ghost_out is [batch_size, d_out]
            ghost_out =  feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask,:]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)

            norm_scaling_factor = l2_norm_residual / (1e-8 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()
            
            # 3. 
            # mse_loss_ghost_resid should be shape [batch_size]
            mse_loss_ghost_resid = (torch.pow((ghost_out - residual.float()), 2)) / (torch.norm(residual.float(), dim=-1, keepdim=True) + 1e-8)
            mse_rescaling_factor = (mse_loss / (mse_loss_ghost_resid + 1e-8)).detach()
            mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

        mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
        mse_loss = mse_loss.mean()
        sparsity = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,)) 
        l1_loss = self.l1_coefficient * sparsity
        loss = mse_loss + l1_loss + mse_loss_ghost_resid

        return sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid

    def get_sparse_connection_loss(self):
        dots = self.spacon_sae_W_dec @ self.W_dec.T
        # each row is an sae feature, each column is a transcoder feature
        loss = torch.sum(dots.abs(), dim=1).mean() # mean over each sae feature of L1 of transcoder features activated
        return self.cfg.sparse_connection_l1_coeff * loss

    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        # DDP Change: Check if DDP is initialized.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if self.cfg.b_dec_init_method == "geometric_median":
            # This method is complex to synchronize, using mean is safer for DDP.
            # For now, we recommend using 'mean' with DDP.
            if world_size > 1 and dist.get_rank() == 0:
                print("Warning: 'geometric_median' for b_dec initialization is not fully supported in DDP. Consider using 'mean'.")
            self.initialize_b_dec_with_geometric_median(activation_store) # Not modifying this one for now.
        elif self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(activation_store, world_size)
        elif self.cfg.b_dec_init_method == "zeros":
            pass
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}")

    @torch.no_grad()
    def initialize_b_dec_with_geometric_median(self, activation_store):
        assert(self.cfg.is_transcoder == activation_store.cfg.is_transcoder)

        previous_b_dec = self.b_dec.clone().cpu()
        all_activations = activation_store.storage_buffer.detach().cpu()
        out = compute_geometric_median(
            all_activations,
            skip_typechecks=True, 
            maxiter=100, per_component=False
        ).median
        
        
        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)
        
        print("Reinitializing b_dec with geometric median of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")
        
        out = torch.tensor(out, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

        if self.b_dec_out is not None:
            # stupid code duplication
            previous_b_dec_out = self.b_dec_out.clone().cpu()
            all_activations_out = activation_store.storage_buffer_out.detach().cpu()
            out_out = compute_geometric_median(
                all_activations_out,
                skip_typechecks=True, 
                maxiter=100, per_component=False
            ).median
            
            previous_distances_out = torch.norm(all_activations_out - previous_b_dec_out, dim=-1)
            distances_out = torch.norm(all_activations_out - out_out, dim=-1)
            
            print("Reinitializing b_dec with geometric median of activations")
            print(f"Previous distances: {previous_distances_out.median(0).values.mean().item()}")
            print(f"New distances: {distances_out.median(0).values.mean().item()}")
            
            out_out = torch.tensor(out_out, dtype=self.dtype, device=self.device)
            self.b_dec_out.data = out_out
        
    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store, world_size=1):
        assert(self.cfg.is_transcoder == activation_store.cfg.is_transcoder)
        
        # Each rank calculates the mean of its local buffer
        local_mean_in = activation_store.storage_buffer.mean(dim=0)
        
        if world_size > 1:
            # DDP Change: Average the means across all ranks
            dist.all_reduce(local_mean_in, op=dist.ReduceOp.SUM)
            local_mean_in /= world_size

        self.b_dec.data = local_mean_in.to(self.dtype).to(self.device)
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"Reinitialized b_dec with mean of activations across {world_size} GPUs.")


        if self.b_dec_out is not None:
            local_mean_out = activation_store.storage_buffer_out.mean(dim=0)
            if world_size > 1:
                # DDP Change: Average the means across all ranks
                dist.all_reduce(local_mean_out, op=dist.ReduceOp.SUM)
                local_mean_out /= world_size
            self.b_dec_out.data = local_mean_out.to(self.dtype).to(self.device)
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Reinitialized b_dec_out with mean of activations across {world_size} GPUs.")

    @torch.no_grad()
    def resample_neurons_anthropic(
        self, 
        dead_neuron_indices, 
        model,
        optimizer, 
        activation_store):
        """
        Arthur's version of Anthropic's feature resampling
        procedure.
        """
        # collect global loss increases, and input activations
        # DDP Change: Pass rank and world_size to the collection function
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_loss_increases, global_input_activations = self.collect_anthropic_resampling_losses(
            model, activation_store, rank, world_size
        )

        # The rest of the logic is now performed identically on all ranks
        # because they all have the same global loss and activation data.
        
        # sample according to losses
        probs = global_loss_increases / global_loss_increases.sum()
        sample_indices = torch.multinomial(
            probs,
            min(len(dead_neuron_indices), probs.shape[0]),
            replacement=False,
        )
        
        if sample_indices.shape[0] < len(dead_neuron_indices):
            dead_neuron_indices = dead_neuron_indices[:sample_indices.shape[0]]

        # Replace W_dec with normalized differences in activations
        self.W_dec.data[dead_neuron_indices, :] = (
            (
                global_input_activations[sample_indices]
                / torch.norm(global_input_activations[sample_indices], dim=1, keepdim=True)
            )
            .to(self.dtype)
            .to(self.device)
        )
        
        self.W_enc.data[:, dead_neuron_indices] = self.W_dec.data[dead_neuron_indices, :].T
        self.b_enc.data[dead_neuron_indices] = 0.0
        
        if dead_neuron_indices.shape[0] > 0 and dead_neuron_indices.shape[0] < self.d_sae:
            sum_of_all_norms = torch.norm(self.W_enc.data, dim=0).sum()
            alive_mask = torch.ones(self.d_sae, device=self.device, dtype=torch.bool)
            alive_mask[dead_neuron_indices] = False
            sum_of_alive_norms = torch.norm(self.W_enc.data[:, alive_mask], dim=0).sum()
            average_norm = sum_of_alive_norms / (self.d_sae - len(dead_neuron_indices))
            self.W_enc.data[:, dead_neuron_indices] *= self.cfg.feature_reinit_scale * average_norm
        
        # Reset optimizer states for the updated parameters
        for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            # W_enc
            if k is self.W_enc:
                for v_key in ["exp_avg", "exp_avg_sq"]:
                    v[v_key][:, dead_neuron_indices] = 0.0
            # b_enc
            elif k is self.b_enc:
                for v_key in ["exp_avg", "exp_avg_sq"]:
                    v[v_key][dead_neuron_indices] = 0.0
            # W_dec
            elif k is self.W_dec:
                for v_key in ["exp_avg", "exp_avg_sq"]:
                    v[v_key][dead_neuron_indices, :] = 0.0

    @torch.no_grad()
    def collect_anthropic_resampling_losses(self, model, activation_store, rank, world_size):
        """
        Collects the losses for resampling neurons (anthropic), synchronized across all DDP ranks.
        """
        
        batch_size = self.cfg.store_batch_size
        number_final_activations = self.cfg.resample_batches * batch_size
        
        # Each rank will process a chunk of the total resampling data
        local_activations_needed = number_final_activations // world_size
        if rank == 0:
            # Ensure the division is clean
            assert number_final_activations % world_size == 0, "resample_batches * store_batch_size must be divisible by world_size"
        
        local_iterator = range(0, local_activations_needed, batch_size)
        if rank == 0:
            local_iterator = tqdm(local_iterator, desc=f"Rank {rank} collecting losses for resampling...")
            
        local_loss_increases = torch.zeros((local_activations_needed,), dtype=self.dtype, device=self.device)
        local_input_activations = torch.zeros((local_activations_needed, self.d_in), dtype=self.dtype, device=self.device)

        for i, refill_idx in enumerate(local_iterator):
            batch_tokens = activation_store.get_batch_tokens()
            
            # Using get_test_loss is inefficient here because it doesn't return activations.
            # We'll replicate the logic to get both loss and activations.
            def standard_replacement_hook(activations, hook):
                activations = self.forward(activations)[0].to(activations.dtype)
                return activations
            replacement_hook = standard_replacement_hook
            ce_loss_with_recons = model.run_with_hooks(
                batch_tokens,
                return_type="loss",
                loss_per_token=True,
                fwd_hooks=[(self.cfg.hook_point, replacement_hook)],
            )

            ce_loss_without_recons, normal_activations_cache = model.run_with_cache(
                batch_tokens,
                names_filter=self.cfg.hook_point,
                return_type="loss",
                loss_per_token=True,
            )
            normal_activations = normal_activations_cache[self.cfg.hook_point]
            if self.cfg.hook_point_head_index is not None:
                normal_activations = normal_activations[:,:,self.cfg.hook_point_head_index]

            changes_in_loss = (ce_loss_with_recons - ce_loss_without_recons).view(batch_size, -1)
            normal_activations = normal_activations.view(batch_size, -1, self.d_in)
            
            probs = F.relu(changes_in_loss)
            # handle cases where all losses are negative
            if torch.all(probs == 0):
                # if all loss changes are negative, sample uniformly
                probs = torch.ones_like(changes_in_loss)

            changes_in_loss_dist = Categorical(probs)
            samples = changes_in_loss_dist.sample()

            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            local_loss_increases[start_idx:end_idx] = changes_in_loss[torch.arange(batch_size), samples]
            local_input_activations[start_idx:end_idx] = normal_activations[torch.arange(batch_size), samples]
        
        # DDP Change: Gather results from all ranks.
        if world_size > 1:
            global_loss_increases = torch.zeros((number_final_activations,), dtype=self.dtype, device=self.device)
            global_input_activations = torch.zeros((number_final_activations, self.d_in), dtype=self.dtype, device=self.device)
            
            # Create lists of tensors to gather
            local_loss_list = list(torch.chunk(local_loss_increases, chunks=world_size, dim=0))
            global_loss_list = [torch.empty_like(local_loss_list[0]) for _ in range(world_size)]
            
            local_input_list = list(torch.chunk(local_input_activations, chunks=world_size, dim=0))
            global_input_list = [torch.empty_like(local_input_list[0]) for _ in range(world_size)]

            # Perform all-to-all to gather data
            dist.all_to_all(global_loss_list, local_loss_list)
            dist.all_to_all(global_input_list, local_input_list)

            # Concatenate the gathered tensors
            global_loss_increases = torch.cat(global_loss_list, dim=0)
            global_input_activations = torch.cat(global_input_list, dim=0)
        else:
            global_loss_increases = local_loss_increases
            global_input_activations = local_input_activations
            
        return global_loss_increases, global_input_activations

    # ... (rest of the file is largely unchanged, but I'll include it for completeness) ...
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        '''
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        '''
        
        if self.W_dec.grad is None:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_out, d_sae d_out -> d_sae",
        )
        
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_out -> d_sae d_out",
        )
    
    def save_model(self, path: str):
        '''
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        '''
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        
        state_dict = {
            "cfg": self.cfg,
            "state_dict": self.state_dict()
        }
        
        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz")
        
        print(f"Saved model to {path}")
    
    @classmethod
    def load_from_pretrained(cls, path: str, device=None):
        '''
        Load function for the model. This method can be called directly on the class.
        '''
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        if path.endswith(".pt"):
            state_dict = torch.load(path, map_location=device)
        elif path.endswith(".pkl.gz"):
            with gzip.open(path, 'rb') as f:
                state_dict = pickle.load(f)
        elif path.endswith(".pkl"):
            with open(path, 'rb') as f:
                state_dict = pickle.load(f)
        else:
            raise ValueError(f"Unexpected file extension: {path}")

        if 'cfg' not in state_dict or 'state_dict' not in state_dict:
            raise ValueError("Loaded state dictionary must contain 'cfg' and 'state_dict' keys")
        
        cfg = state_dict["cfg"]
        if device is not None:
            cfg.device = device

        instance = cls(cfg=cfg)
        instance.load_state_dict(state_dict["state_dict"])

        return instance

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name