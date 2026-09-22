from __future__ import annotations

import torch
from babel.channels.symbolic import (
    CLAIM_TYPES,
    SYMBOLIC_CAPACITY_BITS,
    SymbolicChannel,
    SymbolicMessage,
)


def test_symbolic_capacity_bits_returns_exactly_8() -> None:
    channel = SymbolicChannel(agent_intent_dim=128, observation_dim=313, message_dim=16)

    assert channel.capacity_bits() == 8.0
    assert SYMBOLIC_CAPACITY_BITS == 8


def test_symbolic_encoder_produces_valid_bit_allocated_message() -> None:
    torch.manual_seed(7)
    channel = SymbolicChannel(agent_intent_dim=128, observation_dim=313, message_dim=16)
    agent_intent = torch.randn(128)
    obs = torch.randn(313)

    message = channel.encode(agent_intent, obs, agent_id=2)

    assert isinstance(message, SymbolicMessage)
    assert 0 <= message.claim_type < len(CLAIM_TYPES)
    assert 0 <= message.node_id < 16
    assert 0 <= message.payload < 2
    bits = message.to_bits()
    assert len(bits) == 8
    assert all(bit in {0, 1} for bit in bits)
    assert SymbolicMessage.from_bits(bits) == message


def test_symbolic_decoder_is_deterministic_and_round_trips_bits() -> None:
    channel = SymbolicChannel(agent_intent_dim=128, observation_dim=313, message_dim=16)
    message = SymbolicMessage(claim_type=5, node_id=11, payload=1)
    receiver_obs = torch.randn(313)

    decoded_a = channel.decode(message, receiver_obs)
    decoded_b = channel.decode(SymbolicMessage.from_bits(message.to_bits()), receiver_obs)

    assert decoded_a.shape == (16,)
    assert torch.allclose(decoded_a, decoded_b)


def test_symbolic_batch_sample_outputs_embeddings_and_logprobs() -> None:
    torch.manual_seed(13)
    channel = SymbolicChannel(agent_intent_dim=128, observation_dim=313, message_dim=16)
    agent_intents = torch.randn(6, 128)
    observations = torch.randn(6, 313)

    batch = channel.sample_batch(agent_intents, observations)

    assert batch.slots.shape == (6, 3)
    assert batch.embeddings.shape == (6, 16)
    assert batch.logprobs.shape == (6,)
    assert batch.entropy.shape == (6,)
    assert torch.all((batch.slots[:, 0] >= 0) & (batch.slots[:, 0] < 8))
    assert torch.all((batch.slots[:, 1] >= 0) & (batch.slots[:, 1] < 16))
    assert torch.all((batch.slots[:, 2] >= 0) & (batch.slots[:, 2] < 2))


def test_latent_vq_channel_passes_sanity() -> None:
    from babel.channels.latent_vq import LatentVQChannel, VQMessage

    torch.manual_seed(42)
    channel = LatentVQChannel(
        agent_intent_dim=128,
        observation_dim=313,
        message_dim=16,
        hidden_dim=128,
        latent_dim=64,
        codebook_size=1024,
    )

    agent_intent = torch.randn(128)
    obs = torch.randn(313)

    # Test encode
    message = channel.encode(agent_intent, obs, agent_id=0)
    assert isinstance(message, VQMessage)
    assert 0 <= message.code < 1024
    assert message.embedding.shape == (64,)

    # Test decode
    receiver_obs = torch.randn(313)
    decoded = channel.decode(message, receiver_obs)
    assert decoded.shape == (16,)

    # Test sample_batch
    agent_intents = torch.randn(6, 128)
    observations = torch.randn(6, 313)
    batch = channel.sample_batch(agent_intents, observations)
    assert batch.slots.shape == (6, 1)
    assert batch.embeddings.shape == (6, 16)
    assert batch.logprobs.shape == (6,)
    assert batch.entropy.shape == (6,)
    assert channel.get_loss().shape == ()


def test_nl_channel_passes_sanity() -> None:
    from babel.channels.nl import NLChannel, NLMessage

    torch.manual_seed(42)
    # Use gpt2 as lightweight local model
    channel = NLChannel(
        agent_intent_dim=128,
        observation_dim=313,
        message_dim=16,
        hidden_dim=128,
        model_name="gpt2",
        num_soft_tokens=4,
        max_new_tokens=4,
    )

    agent_intent = torch.randn(128)
    obs = torch.randn(313)

    # Test encode
    message = channel.encode(agent_intent, obs, agent_id=0)
    assert isinstance(message, NLMessage)
    assert isinstance(message.text, str)
    assert message.embedding.shape == (16,)

    # Test decode
    receiver_obs = torch.randn(313)
    decoded = channel.decode(message, receiver_obs)
    assert decoded.shape == (16,)
    assert torch.allclose(decoded, message.embedding)

    # Test sample_batch
    agent_intents = torch.randn(6, 128)
    observations = torch.randn(6, 313)
    batch = channel.sample_batch(agent_intents, observations)
    assert batch.slots.shape == (6, 1)
    assert batch.embeddings.shape == (6, 16)
    assert batch.logprobs.shape == (6,)
    assert batch.entropy.shape == (6,)
