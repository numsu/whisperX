import numpy as np

from whisperx.diarize import DiarizationPipeline, _speaker_embeddings_from_clusters


class FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    def itertracks(self, yield_label=False):
        assert yield_label is True
        return [
            (FakeSegment(0.0, 1.0), "track", 0),
            (FakeSegment(1.0, 2.0), "track", "SPEAKER_01"),
        ]


def test_speaker_embeddings_from_clusters_are_averaged_by_speaker_label():
    embeddings = np.array(
        [
            [[1.0, 3.0], [10.0, 20.0]],
            [[3.0, 5.0], [30.0, 40.0]],
        ]
    )
    hard_clusters = np.array(
        [
            [0, -2],
            [0, 1],
        ]
    )

    assert _speaker_embeddings_from_clusters(embeddings, hard_clusters) == {
        "SPEAKER_00": [2.0, 4.0],
        "SPEAKER_01": [30.0, 40.0],
    }


def test_diarization_to_dataframe_normalizes_integer_speaker_labels():
    diarize_df = DiarizationPipeline._diarization_to_dataframe(FakeDiarization())

    assert diarize_df["speaker"].tolist() == ["SPEAKER_00", "SPEAKER_01"]
    assert diarize_df["start"].tolist() == [0.0, 1.0]
    assert diarize_df["end"].tolist() == [1.0, 2.0]
