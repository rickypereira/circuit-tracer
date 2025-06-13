import os
import torch
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
        rank=0, world_size=1 # DDP Change: Add rank and world_size
    ):
        self.cfg = cfg
        self.model = model
        
        # DDP Change: Store rank and world_size
        self.rank = rank
        self.world_size = world_size
        
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=True)
        self.iterable_dataset = iter(self.dataset)
        
        # DDP Change: Counter to shard the dataset stream across ranks
        self.dataset_item_index = 0
        
        # check if it's tokenized on rank 0
        if self.rank == 0:
            example_item = next(self.iterable_dataset)
            self.dataset_item_index += 1
            if "tokens" in example_item.keys():
                self.cfg.is_dataset_tokenized = True
                print("Dataset is tokenized! Updating config.")
            elif "text" in example_item.keys():
                self.cfg.is_dataset_tokenized = False
                print("Dataset is not tokenized! Updating config.")
            # Reset iterator after peeking
            self.iterable_dataset = iter(self.dataset)
            self.dataset_item_index = 0

        # DDP Change: Broadcast the tokenization status from rank 0 to all other ranks
        if self.world_size > 1:
            is_tokenized_tensor = torch.tensor(int(self.cfg.is_dataset_tokenized), device=self.cfg.device)
            torch.distributed.broadcast(is_tokenized_tensor, src=0)
            self.cfg.is_dataset_tokenized = bool(is_tokenized_tensor.item())

        if self.cfg.use_cached_activations:
            # Not implementing DDP for cached activations for now, as it's complex.
            if self.world_size > 1:
                raise NotImplementedError("Using cached activations with DDP is not supported in this script.")
        
        if create_dataloader:
            self.storage_buffer_out = None
            # The get_buffer method is now DDP-aware because get_batch_tokens is.
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
        device = self.model.device
        

        batch_tokens = torch.zeros(size=(0, context_size), device=device, dtype=torch.long, requires_grad=False)
        current_batch = []
        current_length = 0
        
        while batch_tokens.shape[0] < batch_size:
            # DDP Change: Each rank pulls from the iterator, but only processes its assigned item.
            while True:
                raw_item = next(self.iterable_dataset)
                # This check ensures each rank gets a unique portion of the data stream
                if self.dataset_item_index % self.world_size == self.rank:
                    self.dataset_item_index += 1
                    break
                self.dataset_item_index += 1

            if not self.cfg.is_dataset_tokenized:
                s = raw_item["text"]
                tokens = self.model.to_tokens(
                    s, 
                    truncate=True, 
                    move_to_device=True,
                ).squeeze(0)
                assert len(tokens.shape) == 1, f"tokens.shape should be 1D but was {tokens.shape}"
            else:
                tokens = torch.tensor(
                    raw_item["tokens"],
                    dtype=torch.long,
                    device=device,
                    requires_grad=False,
                )
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
                    batch_tokens = torch.cat((batch_tokens, full_batch.unsqueeze(0)), dim=0)
                    current_batch = []
                    current_length = 0

        return batch_tokens[:batch_size]

    def get_activations(self, batch_tokens, get_loss=False):
        # This method remains unchanged as it's just a model forward pass.
        assert(not (self.cfg.is_transcoder and (self.cfg.hook_point_head_index is not None)))
        
        act_name = self.cfg.hook_point
        hook_point_layer = self.cfg.hook_point_layer
        if self.cfg.hook_point_head_index is not None:
            activations = self.model.run_with_cache(
                batch_tokens,
                names_filter=act_name,
                stop_at_layer=hook_point_layer+1
            )[1][act_name][:,:,self.cfg.hook_point_head_index]
        else:
            if not self.cfg.is_transcoder:
                activations = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=act_name,
                    stop_at_layer=hook_point_layer+1
                )[1][act_name]
            else:
                cache = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=[act_name, self.cfg.out_hook_point],
                    stop_at_layer=self.cfg.out_hook_point_layer+1
                )[1]
                activations = (cache[act_name], cache[self.cfg.out_hook_point])

        return activations

    def get_buffer(self, n_batches_in_buffer):
        # This method remains largely unchanged. It will now be called on each rank,
        # but because get_batch_tokens provides unique data to each rank, the
        # resulting buffer will also be unique.
        
        gc.collect()
        torch.cuda.empty_cache()
        
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer
        
        if self.cfg.use_cached_activations:
           raise NotImplementedError("use_cached_activations is not supported with DDP.")

        # Display progress bar only for rank 0
        pbar_desc = f"Rank {self.rank} filling buffer"
        refill_iterator = range(0, total_size, batch_size)
        if self.rank == 0:
            refill_iterator = tqdm(refill_iterator, desc=pbar_desc)

        new_buffer = torch.zeros((total_size, context_size, d_in), dtype=self.cfg.dtype, device=self.cfg.device)
        new_buffer_out = None
        if self.cfg.is_transcoder:
            new_buffer_out = torch.zeros((total_size, context_size, self.cfg.d_out), dtype=self.cfg.dtype, device=self.cfg.device)

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
        # This method remains unchanged. It operates on the per-rank storage_buffer,
        # which is already unique for each rank.
        
        batch_size = self.cfg.train_batch_size
        
        if self.cfg.is_transcoder:
            new_buffer, new_buffer_out = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            mixing_buffer = torch.cat([new_buffer, self.storage_buffer])
            mixing_buffer_out = torch.cat([new_buffer_out, self.storage_buffer_out])

            assert(mixing_buffer.shape[0] == mixing_buffer_out.shape[0])
            randperm = torch.randperm(mixing_buffer.shape[0])
            mixing_buffer = mixing_buffer[randperm]
            mixing_buffer_out = mixing_buffer_out[randperm]

            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
            self.storage_buffer_out = mixing_buffer_out[:mixing_buffer_out.shape[0]//2]

            catted_buffers = torch.cat([
                mixing_buffer[mixing_buffer.shape[0]//2:],
                mixing_buffer_out[mixing_buffer.shape[0]//2:]
            ], dim=1)

            dataloader = iter(DataLoader(catted_buffers, batch_size=batch_size, shuffle=True))
        else:
            mixing_buffer = torch.cat([self.get_buffer(self.cfg.n_batches_in_buffer // 2), self.storage_buffer])
            mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]
            
            self.storage_buffer = mixing_buffer[:mixing_buffer.shape[0]//2]
            dataloader = iter(DataLoader(mixing_buffer[mixing_buffer.shape[0]//2:], batch_size=batch_size, shuffle=True))
        
        return dataloader
    
    def next_batch(self):
        # This method remains unchanged.
        try:
            return next(self.dataloader)
        except StopIteration:
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)