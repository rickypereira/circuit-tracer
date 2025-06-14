import os
import torch
import torch.distributed as dist  # Make sure dist is imported
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
import gc

class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs. Modified for DDP.
    """
    def __init__(
        self, cfg, model: HookedTransformer, create_dataloader: bool = True,
        rank=0, world_size=1
    ):
        self.cfg = cfg
        self.model = model
        self.rank = rank
        self.world_size = world_size
        
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True, trust_remote_code=True)
        self.iterable_dataset = iter(self.dataset)
        self.dataset_item_index = 0
        
        # Determine the model's device once and for all.
        self.model_device = next(self.model.parameters()).device
        
        # Check tokenization status on rank 0
        if self.rank == 0:
            example_item = next(self.iterable_dataset)
            self.dataset_item_index += 1
            self.cfg.is_dataset_tokenized = "tokens" in example_item
            print(f"Dataset tokenization status: {self.cfg.is_dataset_tokenized}")
            # Reset iterator
            self.iterable_dataset = iter(self.dataset)
            self.dataset_item_index = 0

        # Broadcast the tokenization status
        if self.world_size > 1:
            # FIX: Use the reliable model_device, not cfg.device
            is_tokenized_tensor = torch.tensor(int(self.cfg.is_dataset_tokenized), device=self.model_device)
            dist.broadcast(is_tokenized_tensor, src=0)
            self.cfg.is_dataset_tokenized = bool(is_tokenized_tensor.item())

        if self.cfg.use_cached_activations:
            if self.world_size > 1:
                raise NotImplementedError("Using cached activations with DDP is not supported in this script.")
        
        if create_dataloader:
            self.storage_buffer_out = None
            if self.cfg.is_transcoder:
                self.storage_buffer, self.storage_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            else:
                self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            self.dataloader = self.get_data_loader()

    def get_batch_tokens(self):
        """
        Streams a batch of tokens from a dataset, ensuring each rank gets unique data.
        """
        batch_size = self.cfg.store_batch_size
        context_size = self.cfg.context_size
        
        # FIX: Use the reliable model_device stored in __init__
        device = self.model_device
        
        batch_tokens = torch.zeros(size=(0, context_size), device=device, dtype=torch.long, requires_grad=False)
        current_batch = []
        current_length = 0
        
        while batch_tokens.shape[0] < batch_size:
            while True:
                raw_item = next(self.iterable_dataset)
                if self.dataset_item_index % self.world_size == self.rank:
                    self.dataset_item_index += 1
                    break
                self.dataset_item_index += 1

            if not self.cfg.is_dataset_tokenized:
                s = raw_item["text"]
                # Step 1: Tell to_tokens NOT to move the tensor. It will be created on the CPU.
                tokens = self.model.to_tokens(s, truncate=True, move_to_device=False).squeeze(0)
                # Step 2: Manually move the tensor to the correct device, which we get from the model's parameters.
                tokens = tokens.to(next(self.model.parameters()).device)
            else:
                tokens = torch.tensor(raw_item["tokens"], dtype=torch.long, device=device, requires_grad=False)

            token_len = tokens.shape[0]
            bos_token_id_tensor = torch.tensor([self.model.tokenizer.bos_token_id], device=tokens.device, dtype=torch.long)
            
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                space_left = context_size - current_length
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break
                else:
                    current_batch.append(tokens[:space_left])
                    tokens = tokens[space_left:]
                    tokens = torch.cat((bos_token_id_tensor, tokens), dim=0)
                    token_len = tokens.shape[0]
                    current_length = context_size

                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                   # --- START: DEBUGGING PRINT STATEMENTS ---
                    # This will print diagnostics from each GPU for the first few batches.
                    # We check the item index to avoid spamming the console.
                    if self.dataset_item_index < (self.world_size * 3): # Prints for the first ~3 items per GPU
                        print(
                            f"\n--- [Rank {self.rank}] Debugging torch.cat --- \n"
                            f"  Culprit 1 (batch_tokens): device={batch_tokens.device}, shape={batch_tokens.shape}\n"
                            f"  Culprit 2 (full_batch):   device={full_batch.device}, shape={full_batch.shape}\n"
                            f"  Other: (device):   device={device}\n"
                        )
                    # --- END: DEBUGGING PRINT STATEMENTS ---
                    batch_tokens = torch.cat((batch_tokens, full_batch.unsqueeze(0)), dim=0)
                    current_batch = []
                    current_length = 0

        return batch_tokens[:batch_size]

    def get_activations(self, batch_tokens, get_loss=False):
        # This method is correct.
        act_name = self.cfg.hook_point
        hook_point_layer = self.cfg.hook_point_layer
        if self.cfg.hook_point_head_index is not None:
            activations = self.model.run_with_cache(batch_tokens, names_filter=act_name, stop_at_layer=hook_point_layer + 1)[1][act_name][:, :, self.cfg.hook_point_head_index]
        else:
            if not self.cfg.is_transcoder:
                activations = self.model.run_with_cache(batch_tokens, names_filter=act_name, stop_at_layer=hook_point_layer + 1)[1][act_name]
            else:
                cache = self.model.run_with_cache(batch_tokens, names_filter=[act_name, self.cfg.out_hook_point], stop_at_layer=self.cfg.out_hook_point_layer + 1)[1]
                activations = (cache[act_name], cache[self.cfg.out_hook_point])
        return activations

    def get_buffer(self, n_batches_in_buffer):
        gc.collect()
        
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer
        
        if self.cfg.use_cached_activations:
           raise NotImplementedError("use_cached_activations is not supported with DDP.")

        pbar_desc = f"Rank {self.rank} filling buffer"
        refill_iterator = range(0, total_size, batch_size)
        if self.rank == 0:
            refill_iterator = tqdm(refill_iterator, desc=pbar_desc)

        # FIX: Use the reliable model_device, not cfg.device
        new_buffer = torch.zeros((total_size, context_size, d_in), dtype=self.cfg.dtype, device=self.model_device)
        new_buffer_out = None
        if self.cfg.is_transcoder:
            new_buffer_out = torch.zeros((total_size, context_size, self.cfg.d_out), dtype=self.cfg.dtype, device=self.model_device)

        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()
            if not self.cfg.is_transcoder:
                refill_activations = self.get_activations(refill_batch_tokens)
                new_buffer[refill_batch_idx_start : refill_batch_idx_start + batch_size] = refill_activations
            else:
                refill_activations_in, refill_activations_out = self.get_activations(refill_batch_tokens)
                new_buffer[refill_batch_idx_start : refill_batch_idx_start + batch_size] = refill_activations_in
                new_buffer_out[refill_batch_idx_start : refill_batch_idx_start + batch_size] = refill_activations_out

        new_buffer = new_buffer.reshape(-1, d_in)
        randperm = torch.randperm(new_buffer.shape[0])
        new_buffer = new_buffer[randperm]

        if self.cfg.is_transcoder:
            new_buffer_out = new_buffer_out.reshape(-1, self.cfg.d_out)
            new_buffer_out = new_buffer_out[randperm]
            return new_buffer, new_buffer_out
        else:
            return new_buffer

    def get_data_loader(self,) -> DataLoader:
        # This method is correct.
        batch_size = self.cfg.train_batch_size
        if self.cfg.is_transcoder:
            new_buffer, new_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            mixing_buffer = torch.cat([new_buffer, self.storage_buffer])
            mixing_buffer_out = torch.cat([new_buffer_out, self.storage_buffer_out])
            assert mixing_buffer.shape[0] == mixing_buffer_out.shape[0]
            randperm = torch.randperm(mixing_buffer.shape[0])
            mixing_buffer = mixing_buffer[randperm]
            mixing_buffer_out = mixing_buffer_out[randperm]
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0] // 2]
            self.storage_buffer_out = mixing_buffer_out[:mixing_buffer_out.shape[0] // 2]
            catted_buffers = torch.cat([mixing_buffer[mixing_buffer.shape[0] // 2:], mixing_buffer_out[mixing_buffer_out.shape[0] // 2:]], dim=1)
            dataloader = iter(DataLoader(catted_buffers, batch_size=batch_size, shuffle=True))
        else:
            mixing_buffer = torch.cat([self.get_buffer(self.cfg.n_batches_in_buffer // 2), self.storage_buffer])
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0] // 2]
            dataloader = iter(DataLoader(mixing_buffer[mixing_buffer.shape[0] // 2:], batch_size=batch_size, shuffle=True))
        return dataloader
    
    def next_batch(self):
        # This method is correct.
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)