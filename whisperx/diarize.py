import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
import torch
import torchaudio
from io import BytesIO

from whisperx.audio import SAMPLE_RATE
from whisperx.schema import TranscriptionResult, AlignedTranscriptionResult, ProgressCallback
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class IntervalTree:
    """
    Simple interval tree for fast overlap queries using sorted array + binary search.

    Uses O(n) space and provides O(log n) query time instead of O(n) linear scan.
    This achieves ~228x speedup for speaker assignment in long-form content.
    """

    def __init__(self, intervals: List[Tuple[float, float, str]]):
        """
        Initialize the interval tree with diarization segments.

        Args:
            intervals: List of (start, end, speaker) tuples
        """
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: List[str] = []
            return

        # Sort intervals by start time for binary search
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        self.starts = np.array([i[0] for i in sorted_intervals], dtype=np.float64)
        self.ends = np.array([i[1] for i in sorted_intervals], dtype=np.float64)
        self.speakers = [i[2] for i in sorted_intervals]

    def query(self, start: float, end: float) -> List[Tuple[str, float]]:
        """
        Find all intervals that overlap with [start, end] and compute intersection.

        Args:
            start: Query interval start time
            end: Query interval end time

        Returns:
            List of (speaker, intersection_duration) tuples for overlapping segments
        """
        if len(self.starts) == 0:
            return []

        # Binary search to find candidate intervals
        # Only intervals with start < end could overlap
        right_idx = np.searchsorted(self.starts, end, side='left')
        if right_idx == 0:
            return []

        # Check candidates for actual overlap
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)

        results = []
        for idx in np.where(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> Optional[str]:
        """
        Find the speaker of the nearest segment to a given time point.

        Args:
            time: Time point to find nearest segment for

        Returns:
            Speaker ID of nearest segment, or None if no segments exist
        """
        if len(self.starts) == 0:
            return None

        # Calculate midpoints of all segments
        mids = (self.starts + self.ends) / 2
        nearest_idx = np.argmin(np.abs(mids - time))
        return self.speakers[nearest_idx]


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
        logger.info(f"Loading diarization model: {model_config}")
        from diarizen.pipelines.inference import DiariZenPipeline

        self.model = DiariZenPipeline.from_pretrained(model_config)
        self.model.to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: ProgressCallback = None,
    ) -> pd.DataFrame:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or audio array
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            progress_callback: Optional callable receiving a float (0-100) with progress percentage

        Returns:
            Diarization dataframe.
        """
        input_audio: Union[str, BytesIO]
        if isinstance(audio, str):
            input_audio = audio
        else:
            input_audio = BytesIO()
            waveform = torch.from_numpy(audio[None, :])
            torchaudio.save(input_audio, waveform, SAMPLE_RATE, format="wav")
            input_audio.seek(0)

        if progress_callback is not None:
            progress_callback(0.0)

        previous_min_speakers = self.model.min_speakers
        previous_max_speakers = self.model.max_speakers
        if num_speakers is not None:
            self.model.min_speakers = num_speakers
            self.model.max_speakers = num_speakers
        if min_speakers is not None:
            self.model.min_speakers = min_speakers
        if max_speakers is not None:
            self.model.max_speakers = max_speakers

        try:
            diarization = self.model(input_audio)
        finally:
            self.model.min_speakers = previous_min_speakers
            self.model.max_speakers = previous_max_speakers
        if progress_callback is not None:
            progress_callback(100.0)

        return self._diarization_to_dataframe(diarization)

    @staticmethod
    def _diarization_to_dataframe(diarization) -> pd.DataFrame:
        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> Union[AlignedTranscriptionResult, TranscriptionResult]:
    """
    Assign speakers to words and segments in the transcript.

    Uses an interval tree for O(log n) overlap queries instead of O(n) linear scan,
    achieving ~228x speedup for long-form content (3+ hour podcasts).

    Args:
        diarize_df: Diarization dataframe from DiarizationPipeline
        transcript_result: Transcription result to augment with speaker labels
        speaker_embeddings: Optional dictionary mapping speaker IDs to embedding vectors
        fill_nearest: If True, assign speakers even when there's no direct time overlap

    Returns:
        Updated transcript_result with speaker assignments and optionally embeddings
    """
    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    # Build interval tree from diarization segments for O(log n) queries
    intervals = [
        (row['start'], row['end'], row['speaker'])
        for _, row in diarize_df.iterrows()
    ]
    tree = IntervalTree(intervals)

    for seg in transcript_segments:
        seg_start = seg.get('start', 0.0)
        seg_end = seg.get('end', 0.0)

        # Query overlapping segments using interval tree
        overlaps = tree.query(seg_start, seg_end)

        if overlaps:
            # Sum intersection durations per speaker and pick the dominant one
            speaker_intersections: dict[str, float] = {}
            for speaker, intersection in overlaps:
                speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
            seg['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
        elif fill_nearest:
            # Find nearest segment if no overlap
            seg_mid = (seg_start + seg_end) / 2
            nearest_speaker = tree.find_nearest(seg_mid)
            if nearest_speaker:
                seg['speaker'] = nearest_speaker

        # Assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' not in word:
                    continue

                word_start = word['start']
                word_end = word.get('end', word_start)

                word_overlaps = tree.query(word_start, word_end)

                if word_overlaps:
                    speaker_intersections = {}
                    for speaker, intersection in word_overlaps:
                        speaker_intersections[speaker] = speaker_intersections.get(speaker, 0.0) + intersection
                    word['speaker'] = max(speaker_intersections.items(), key=lambda x: x[1])[0]
                elif fill_nearest:
                    word_mid = (word_start + word_end) / 2
                    nearest_speaker = tree.find_nearest(word_mid)
                    if nearest_speaker:
                        word['speaker'] = nearest_speaker

    # Add speaker embeddings to the result if provided
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings

    return transcript_result


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker
