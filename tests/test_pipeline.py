import torch

from torch_3d_tracker.pipeline import TrackerPipeline


def test_depth_and_features_are_generated():
    pipeline = TrackerPipeline()
    image = torch.rand(1, 3, 64, 64)
    detections = [torch.tensor([[10.0, 10.0, 30.0, 30.0]])]

    result = pipeline(image, detections=detections)

    assert result.depth_map.shape == (1, 1, 64, 64)
    assert result.spatial_features.shape[0] == 1
    assert result.spatial_features.shape[1] == 16
    assert len(result.tracks) >= 1


def test_demo_sequence_runs():
    pipeline = TrackerPipeline()
    sequence = pipeline.generate_demo_sequence(num_frames=2, height=48, width=48)
    assert len(sequence) == 2
    assert sequence[0].shape == (1, 3, 48, 48)
