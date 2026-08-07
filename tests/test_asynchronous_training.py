from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from model.asynchronous_module.runtime import (
    ActorPolicy,
    InferenceRequest,
    InferenceResponse,
    RemoteESM2Encoder,
    run_esm_inference_worker,
)
from model.encoding_module.esm2_encoder import ESM2SequenceEncoder


class _FakeESMModel:
    def __call__(self, tokens, *, repr_layers):
        embedding_dim = 4
        values = tokens.to(dtype=torch.float32).unsqueeze(-1)
        representations = values.repeat(1, 1, embedding_dim)
        return {"representations": {repr_layers[0]: representations}}


def _fake_batch_converter(records):
    max_length = max(len(sequence) for _, sequence in records)
    tokens = torch.zeros((len(records), max_length + 2), dtype=torch.long)
    for row, (_, sequence) in enumerate(records):
        tokens[row, 0] = 1
        tokens[row, 1 : len(sequence) + 1] = torch.arange(2, len(sequence) + 2)
        tokens[row, len(sequence) + 1] = 2
    return None, None, tokens


def test_esm2_encoder_batches_variable_length_sequences() -> None:
    encoder = ESM2SequenceEncoder.__new__(ESM2SequenceEncoder)
    encoder.model = _FakeESMModel()
    encoder.batch_converter = _fake_batch_converter
    encoder.embedding_dim = 4
    encoder.representation_layer = 1
    encoder.device = torch.device("cpu")

    short, long = encoder.encode_sequences(["AC", "ACDE"])

    assert short.shape == (2, 4)
    assert long.shape == (4, 4)
    np.testing.assert_array_equal(short[:, 0], np.asarray([2.0, 3.0]))
    np.testing.assert_array_equal(long[:, 0], np.asarray([2.0, 3.0, 4.0, 5.0]))


def test_remote_encoder_round_trip() -> None:
    requests: queue.Queue = queue.Queue()
    responses: queue.Queue = queue.Queue()
    encoder = RemoteESM2Encoder(
        client_id=3,
        request_queue=requests,
        response_queue=responses,
        timeout_seconds=1.0,
    )

    def respond() -> None:
        request = requests.get(timeout=1.0)
        responses.put(
            InferenceResponse(
                request.request_id,
                embedding=np.ones((len(request.sequence), 4), dtype=np.float32),
            )
        )

    thread = threading.Thread(target=respond)
    thread.start()
    result = encoder.encode_sequence("ACD")
    thread.join(timeout=1.0)

    assert result.shape == (3, 4)
    assert result.dtype == np.float32


class _RecordingBatchEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_sequences(self, sequences):
        self.batch_sizes.append(len(sequences))
        return [
            np.full((len(sequence), 4), len(sequence), dtype=np.float32)
            for sequence in sequences
        ]


def test_inference_worker_dynamically_batches_requests() -> None:
    requests: queue.Queue = queue.Queue()
    responses = [queue.Queue(), queue.Queue(), queue.Queue()]
    ready: queue.Queue = queue.Queue()
    stop = threading.Event()
    recording_encoder = _RecordingBatchEncoder()

    for client_id, sequence in enumerate(("AC", "ACD", "ACDE")):
        requests.put(InferenceRequest(str(client_id), client_id, sequence))

    thread = threading.Thread(
        target=run_esm_inference_worker,
        kwargs={
            "worker_id": 0,
            "device": "cpu",
            "embedding_dim": 4,
            "model_dir": None,
            "request_queue": requests,
            "response_queues": responses,
            "ready_queue": ready,
            "stop_event": stop,
            "batch_size": 8,
            "batch_wait_ms": 50.0,
            "encoder_factory": lambda **_: recording_encoder,
        },
    )
    thread.start()
    assert ready.get(timeout=1.0)["kind"] == "esm_ready"

    returned = [response.get(timeout=1.0) for response in responses]
    requests.put(None)
    thread.join(timeout=1.0)

    assert recording_encoder.batch_sizes == [3]
    assert [item.embedding.shape for item in returned] == [(2, 4), (3, 4), (4, 4)]


def test_actor_policy_respects_variable_length_action_mask() -> None:
    source = ActorPolicy(
        embedding_dim=1280,
        hidden_dims=(2,),
        state_dict=_policy_state_with_increasing_output_bias(),
        seed=3,
    )
    state = np.zeros((3, 1280), dtype=np.float32)
    mask = np.zeros(3 * 20, dtype=np.bool_)
    mask[4] = True
    mask[27] = True

    action = source.select_action(state, mask, epsilon=0.0)

    assert action == 27


def _policy_state_with_increasing_output_bias():
    from model.agent_module.ddqn_agent import QNetwork

    network = QNetwork(
        (1, 1280),
        20,
        hidden_dims=(2,),
        embedding_dim=1280,
    )
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.residue_head[-1].bias.copy_(torch.arange(20, dtype=torch.float32))
    return network.state_dict()
