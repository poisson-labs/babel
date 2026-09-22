from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from babel.channels.base import ChannelModule


@dataclass(frozen=True, slots=True)
class NLMessage:
    text: str
    embedding: Tensor


@dataclass(frozen=True, slots=True)
class NLBatch:
    slots: Tensor  # shape (batch_size, 1), dummy slots representing message hashes or token IDs
    embeddings: Tensor  # shape (batch_size, message_dim), the decoded message embeddings
    logprobs: Tensor  # shape (batch_size,), dummy logprobs (zeros)
    entropy: Tensor  # shape (batch_size,), dummy entropy (zeros)


def _get_input_embeddings(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_input_embeddings"):
        return cast(nn.Module, model.get_input_embeddings())
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return cast(nn.Module, model.transformer.wte)
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return cast(nn.Module, model.model.embed_tokens)
    else:
        raise AttributeError("Could not find input embeddings layer on the LLM model.")


class NLChannel(nn.Module, ChannelModule[NLMessage]):
    def __init__(
        self,
        *,
        agent_intent_dim: int,
        observation_dim: int,
        message_dim: int = 16,
        hidden_dim: int = 128,
        model_name: str = "gpt2",
        num_soft_tokens: int = 8,
        max_new_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.agent_intent_dim = agent_intent_dim
        self.observation_dim = observation_dim
        self.message_dim = message_dim
        self.model_name = model_name
        self.num_soft_tokens = num_soft_tokens
        self.max_new_tokens = max_new_tokens

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size

        input_dim = agent_intent_dim + observation_dim
        self.sender_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_soft_tokens * self.hidden_size),
        )

        self.receiver_projection = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
        )

    def encode(self, agent_intent: Tensor, obs: Tensor, agent_id: int) -> NLMessage:
        del agent_id
        was_batched = agent_intent.ndim > 1
        if not was_batched:
            agent_intent = agent_intent.unsqueeze(0)
            obs = obs.unsqueeze(0)

        device = agent_intent.device
        self.llm = self.llm.to(device)
        embed_layer = _get_input_embeddings(self.llm)

        # 1. Project to soft token embeddings
        soft_embeddings = self.sender_adapter(torch.cat([agent_intent, obs], dim=-1))
        soft_embeddings = soft_embeddings.view(-1, self.num_soft_tokens, self.hidden_size)

        # 2. Tokenize prompt and get prompt embeddings
        prompt_text = "Agent broadcast: "
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeddings = embed_layer(prompt_ids)
        prompt_embeddings_batched = prompt_embeddings.expand(agent_intent.shape[0], -1, -1)

        # 3. Concatenate and generate text autoregressively
        inputs_embeds = torch.cat([soft_embeddings, prompt_embeddings_batched], dim=1)

        with torch.no_grad():
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 4. Receiver projection on the generated text
        # Tokenize the generated text back to embeddings
        tokens = self.tokenizer(decoded_text, return_tensors="pt", padding=True).to(device)
        gen_embeds = embed_layer(tokens.input_ids)
        outputs = self.llm(inputs_embeds=gen_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        embedding = self.receiver_projection(last_hidden)[0]

        return NLMessage(text=decoded_text, embedding=embedding)

    def decode(self, message: NLMessage, receiver_obs: Tensor) -> Tensor:
        del receiver_obs
        # Return the pre-computed embedding from the message
        return message.embedding

    def sample_batch(self, agent_intents: Tensor, observations: Tensor) -> NLBatch:
        # For training, use the fast, fully-differentiable soft-token forward pass path
        device = agent_intents.device
        self.llm = self.llm.to(device)
        embed_layer = _get_input_embeddings(self.llm)
        batch_size = agent_intents.shape[0]

        soft_embeddings = self.sender_adapter(torch.cat([agent_intents, observations], dim=-1))
        soft_embeddings = soft_embeddings.view(batch_size, self.num_soft_tokens, self.hidden_size)

        prompt_text = "Agent broadcast: "
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_embeddings = embed_layer(prompt_ids)
        prompt_embeddings_batched = prompt_embeddings.expand(batch_size, -1, -1)

        inputs_embeds = torch.cat([soft_embeddings, prompt_embeddings_batched], dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        embeddings = self.receiver_projection(last_hidden)
        zeros = torch.zeros(batch_size, dtype=embeddings.dtype, device=device)

        return NLBatch(
            slots=zeros.unsqueeze(-1).long(),
            embeddings=embeddings,
            logprobs=zeros,
            entropy=zeros,
        )

    def deterministic_batch(self, agent_intents: Tensor, observations: Tensor) -> NLBatch:
        # During eval, we can run the differentiable path for efficiency,
        # but the encode/decode methods will run the discrete generation path.
        return self.sample_batch(agent_intents, observations)

    def evaluate_slots(
        self,
        agent_intents: Tensor,
        observations: Tensor,
        slots: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = agent_intents.shape[0]
        zeros = torch.zeros(batch_size, dtype=agent_intents.dtype, device=agent_intents.device)
        return zeros, zeros

    def capacity_bits(self) -> float:
        # NL channel matched capacity is B=8 or B=10
        # Since we estimate Shannon entropy post-hoc, we return the target capacity of 10 bits.
        return 10.0
