from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2Model,
    GPT2LMHeadModel,
)

class CustomizedGPT2Attention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]:
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if past_key_value is not None:
            # Concatenate past key and value tensors for caching
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)  # Concatenate along sequence length
            value = torch.cat([past_value, value], dim=2)

        # Prepare the next set of key and value tensors for caching
        next_past_key_value = (key, value) if use_cache else None

        # Perform the attention operation
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        # Merge the attention heads back together
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # Apply the output projection
        attn_output = self.c_proj(attn_output)
        # Apply dropout to the output
        attn_output = self.resid_dropout(attn_output)

        # Return the attention output and the new key-value pairs (if caching)
        return attn_output, next_past_key_value


class CustomizedGPT2Block(GPT2Block):

    def __init__(self, config, layer_idx=None):
        # Initialize the parent GPT2Block class
        super().__init__(config, layer_idx=layer_idx)
        # Replace the attention module with the customized version
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor]]]:
        # Save the residual for the first residual connection
        residual = hidden_states

        # Apply layer normalization before attention
        hidden_states = self.ln_1(hidden_states)

        # Perform attention with optional caching
        attn_output, next_past_key_value = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # First residual connection
        hidden_states = attn_output + residual

        # Save the residual for the second residual connection
        residual = hidden_states

        # Apply layer normalization before the feed-forward network
        hidden_states = self.ln_2(hidden_states)
        # Apply the feed-forward network
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Second residual connection
        hidden_states = residual + feed_forward_hidden_states

        # Return the hidden states and the new key-value pairs (if caching)
        if use_cache:
            return hidden_states, next_past_key_value
        else:
            return hidden_states, None


class CustomizedGPT2Model(GPT2Model):
    """
    Customized GPT-2 model.

    This class replaces the standard transformer blocks with customized blocks.
    """

    def __init__(self, config):
        # Initialize the parent GPT2Model class
        super().__init__(config)
        # Replace standard blocks with customized blocks
        self.h = nn.ModuleList(
            [CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation


        # Post-initialization processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        batch_size, _ = input_ids.shape

        # Compute input embeddings
        inputs_embeds = self.wte(input_ids)  # Word embeddings
        # Create position IDs by cumulatively summing the attention mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        # Replace padding positions with a default position ID
        position_ids.masked_fill_(attention_mask == 0, 1)
        # Compute position embeddings
        position_embeds = self.wpe(position_ids)
        # Combine word and position embeddings
        hidden_states = inputs_embeds + position_embeds

        # Prepare the attention mask for use in attention layers
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]  # Expand dimensions for broadcasting
        attention_mask = attention_mask.to(dtype=self.dtype)  # Ensure correct data type
        # Convert attention mask to large negative values where mask is 0
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Apply dropout to the embeddings
        hidden_states = self.drop(hidden_states)

        # Define the output shape for the hidden states
        output_shape = (-1, 1, hidden_states.size(-1))
        # Keep only the last time step (useful for generation)
        hidden_states = hidden_states[:, -1:]

        if past_key_values is None:
            # Initialize past_key_values if not provided
            past_key_values = [None] * len(self.h)

        # Prepare to collect new past key-value pairs if caching is enabled
        next_past_key_values = [] if use_cache else None

        # Iterate over transformer blocks
        for i, (block, past_key_value) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            # Update hidden states
            hidden_states = outputs[0]
            if use_cache:
                # Collect new past key-value pairs
                next_past_key_values.append(outputs[1])

        # Apply final layer normalization
        hidden_states = self.ln_f(hidden_states)
        # Reshape the hidden states to the output shape
        hidden_states = hidden_states.view(output_shape)

        # Prepare the output dictionary
        if not use_cache:
            return {'hidden_states': hidden_states}
        else:
            return {
                'hidden_states': hidden_states,
                'past_key_values': tuple(next_past_key_values),
            }


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    """
    Customized GPT-2 Language Modeling Head Model.

    This class replaces the standard GPT-2 model with the customized version.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # Initialize the parent GPT2LMHeadModel class
        super().__init__(config)
        # Replace the transformer with the customized model
        self.transformer = CustomizedGPT2Model(config)

        # Post-initialization processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
    ):
        # Pass inputs through the customized transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Extract hidden states and new past key-value pairs
        hidden_states = outputs["hidden_states"]
        new_past_key_values = outputs["past_key_values"] if use_cache else None

        # Compute logits using the language modeling head
        lm_logits = self.lm_head(hidden_states)

        # Return the logits and new past key-value pairs
        return {
            'logits': lm_logits,
            'past_key_values': new_past_key_values,
        }
