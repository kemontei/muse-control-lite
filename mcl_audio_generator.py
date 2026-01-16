import threading
from queue import Queue
import soundfile as sf
import numpy as np
#import pandas as pd
from pathlib import Path
import sys

sys.path.append('../audio_generator/binaural-beats/src')
from binaural_beats import BinauralBeatGenerator

#N_TRACKS_IN_BUCKET = 1 # Number of tracks to consider in each bucket

# class AudioQueue():
#     def __init__(self):
#         self.queue = Queue()
#         self.last_track = None

#     def put_audio(self, audio_path):
#         self.queue.put(audio_path)
#         self.last_track = audio_path

#     def get_audio(self):
#         return self.queue.get()
    
#     def empty(self):
#         return self.queue.empty()

class MCLAudioGenerator():
    def __init__(self, sample_rate, base_freq, beat_freq, buffer_size, audio_dir):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        #self.bucket = None
        #self.update_audio_bucket(beat_freq)

        # self.audio_dir = Path(audio_dir)
        # if not self.audio_dir.is_dir():
        #     raise Exception # TODO add error details
        # if not (self.audio_dir / "audio_scores.csv").is_file():
        #     raise Exception

        self.bb_gen = BinauralBeatGenerator(sample_rate, base_freq, beat_freq, buffer_size)

    def update_audio_bucket(self, beat_freq):
        if beat_freq >= 1 and beat_freq < 4:
            self.bucket = "delta"
        elif beat_freq >= 4 and beat_freq < 8:
            self.bucket = "theta"
        elif beat_freq >= 8 and beat_freq <= 14:
            self.bucket = "alpha"
        else:
            print("Binaural beat freq is not in a recognized range")
        print(f"Set bucket to {self.bucket}")
        return

    def load_audio(self, file_path, sample_rate):
        audio, sr = sf.read(file_path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"Sample rate mismatch: file is {sr}, expected {sample_rate}")
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        return audio

    def audio_generator(self):
        gen = self.bb_gen.audio_generator()

        audio_scores_df = pd.read_csv(str(self.audio_dir / "audio_scores.csv"))

        while True:
            audio_scores_df = audio_scores_df.sort_values(by="score_"+self.bucket, ascending=False).reset_index(drop=True)
            audio_top_n_scores_df = audio_scores_df.head(N_TRACKS_IN_BUCKET)
            
            # Select 1 row from the top n audio tracks
            audio_file = audio_top_n_scores_df.sample(n=1).loc[0, "file"]
            #audio_file = audio_scores_df.loc[0, "file"]
            print(f"Loading audio file {str(self.audio_dir / audio_file)}")
            audio_data = self.load_audio(str(self.audio_dir / audio_file), self.sample_rate)

            position = 0
            
            while position < len(audio_data):
                bb_segment_data = np.frombuffer(next(gen), dtype=np.float32).reshape(self.buffer_size, 2)
                
                if position + self.buffer_size < len(audio_data):
                    audio_segment_data = audio_data[position:position+self.buffer_size, :]
                else:
                    audio_segment_data = np.zeros((self.buffer_size, 2), dtype=np.float32)
                    remaining = len(audio_data) - position
                    if remaining > 0:
                        audio_segment_data[0:remaining, :] = audio_data[position:position+remaining, :]

                # Mix audio
                mixed_segment_data = 0.9*audio_segment_data + 0.1*bb_segment_data
                
                position += self.buffer_size
                yield mixed_segment_data.tobytes()

    def update_binaural_freq(self, base_freq, beat_freq):
        self.bb_gen.update_binaural_freq(base_freq, beat_freq)


