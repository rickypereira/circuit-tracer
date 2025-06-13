from typing import Tuple

import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder


class LMSparseAutoencoderSessionloader():
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg
        
    # DDP Change: The method now accepts rank and world_size to pass down.
    def load_session(self) -> Tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
        '''
        Loads a session for training a sparse autoencoder on a language model.
        '''
        model_dtype = self.cfg.dtype
        if model_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print(f"Warning: bfloat16 is not supported on this device. Falling back to float16.")
            model_dtype = torch.float16

        model.to(self.cfg.device)

        # Load the model directly onto the correct device in the correct precision.
        model = self.get_model(
            model_name=self.cfg.model_name,
            device=self.cfg.device,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        # DDP Change: Pass rank and world_size to the activations loader.
        activations_loader = self.get_activations_loader(self.cfg, model, self.cfg.rank, self.cfg.world_size)
        sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)
            
        return model, sparse_autoencoder, activations_loader
    
    @classmethod
    # DDP Change: The method now accepts rank and world_size to pass down.
    def load_session_from_pretrained(cls, path: str, rank: int = 0, world_size: int = 1) -> Tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
        '''
        Loads a session for analysing a pretrained sparse autoencoder.
        '''
        if torch.backends.mps.is_available():
            map_location = "mps"
        elif torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
            
        # Ensure device in cfg matches the DDP rank's device
        state = torch.load(path, map_location=map_location)
        cfg = state["cfg"]
        if "cuda" in str(map_location):
             cfg.device = f"cuda:{rank}"
        else:
             cfg.device = map_location

        # DDP Change: Pass rank and world_size to the load_session call.
        model, _, activations_loader = cls(cfg).load_session(rank=rank, world_size=world_size)
        sparse_autoencoder = SparseAutoencoder.load_from_pretrained(path, device=cfg.device)
        
        return model, sparse_autoencoder, activations_loader
    
    def get_model(self, model_name: str, device: str, torch_dtype: torch.dtype):
        '''
        Loads a model from transformer lens directly onto a device in a specific dtype.
        '''
        model = HookedTransformer.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        return model
    
    def initialize_sparse_autoencoder(self, cfg: LanguageModelSAERunnerConfig):
        '''
        Initializes a sparse autoencoder
        '''
        
        sparse_autoencoder = SparseAutoencoder(cfg)
        
        return sparse_autoencoder
    
    # DDP Change: The method now accepts rank and world_size.
    def get_activations_loader(self, cfg: LanguageModelSAERunnerConfig, model: HookedTransformer, rank: int, world_size: int):
        '''
        Loads a DataLoaderBuffer for the activations of a language model.
        '''
        
        # DDP Change: Pass rank and world_size to the ActivationsStore constructor.
        activations_loader = ActivationsStore(
            cfg,
            model,
            rank=rank,
            world_size=world_size
        )
        
        return activations_loader

# This function is for offline processing of cached activations and is not
# affected by the DDP training loop. No changes needed.
def shuffle_activations_pairwise(datapath: str, buffer_idx_range: Tuple[int, int]):
    """
    Shuffles two buffers on disk.
    """
    assert buffer_idx_range[0] < buffer_idx_range[1], \
        "buffer_idx_range[0] must be smaller than buffer_idx_range[1]"
    
    buffer_idx1 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    while buffer_idx1 == buffer_idx2: # Make sure they're not the same
        buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    
    buffer1 = torch.load(f"{datapath}/{buffer_idx1}.pt")
    buffer2 = torch.load(f"{datapath}/{buffer_idx2}.pt")
    joint_buffer = torch.cat([buffer1, buffer2])
    
    # Shuffle them
    joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
    shuffled_buffer1 = joint_buffer[:buffer1.shape[0]]
    shuffled_buffer2 = joint_buffer[buffer1.shape[0]:]
    
    # Save them back
    torch.save(shuffled_buffer1, f"{datapath}/{buffer_idx1}.pt")
    torch.save(shuffled_buffer2, f"{datapath}/{buffer_idx2}.pt")