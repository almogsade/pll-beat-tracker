import numpy as np
from scipy import signal
from scipy import io
import soundfile as sf
from pathlib import Path


class BeatTrackProcessor:
    def __init__(self):
        
        self.signal_type = 'song'
        self.raw_signal = None
        self.sr = 44100
        self.duration = 10

        self.waveform_type = 'sine'
        self.waveform_tempo = 120
        self.waveform_phase = 0.0
        self.waveform_duty = 0.5

        self.song_path = None
        self.song_start = 0
        self.song_stop = 0

        self.onset_win = 0.01
        self.onset_win_type = 'boxcar' # {'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett'}
        self.onset_signal = None

        self.loop_gain = 0.001
        self.vco_tempo = 120
        self.loop_filter = 'order_1'
        self.vco_out = None
        self.inst_tempo = None
        
        self.fix_lag = True
        self.metro_win = 20
        self.metro_signal = None
        self.metro_lag = None

    def update_params(
        self,
        signal_type: str = 'song',
        waveform_type: str = 'sine',
        waveform_tempo: float = 120.0,
        waveform_phase: float = 0.0,
        waveform_duty: float = 0.5,
        duration: float = 10.0,
        song_path: str | None = None,
        song_start: float | None = None,
        song_stop: float | None = None,
        onset_win: float = 0.01,
        onset_win_type: str = 'boxcar',
        loop_gain: float = 0.001,
        vco_tempo: float = 120,
        loop_filter: str = 'order_1',
        fix_lag: bool = True,
        metro_win: int = 20,
    ):
        """Update processor parameters."""
        self.signal_type = signal_type
        self.waveform_type = waveform_type
        self.waveform_tempo = waveform_tempo
        self.waveform_phase = waveform_phase
        self.waveform_duty = waveform_duty
        self.duration = duration
        if song_path is not None: self.song_path = song_path
        if song_start is not None: self.song_start = song_start
        if song_stop is not None: self.song_stop = song_stop
        self.onset_win = onset_win
        self.onset_win_type = onset_win_type
        self.loop_gain = loop_gain
        self.vco_tempo = vco_tempo
        self.loop_filter = loop_filter
        self.fix_lag = fix_lag
        self.metro_win = metro_win

    def get_data(self):
        if self.signal_type == 'waveform':
            self.raw_signal = self.generate_waveform()
        elif self.signal_type == 'song':
            self.raw_signal = self.get_audio_segment()
        else:
            raise KeyError(f"No such signal type: {self.signal_type}")
        
        return self.raw_signal

    def generate_waveform(self) -> np.ndarray:

        t = np.arange(0, self.duration, 1/self.sr)
        frequency = self.waveform_tempo / 60.0

        self.waveform_type = self.waveform_type.lower()

        if self.waveform_type == 'sine':
            wave = np.sin(2 * np.pi * frequency * t + self.waveform_phase)
        elif self.waveform_type == 'square':
            # Duty cycle: proportion of cycle spent at +1
            wave = np.where(((t * frequency + self.waveform_phase/(2*np.pi)) % 1) < self.waveform_duty, 1.0, -1.0)
        elif self.waveform_type == 'triangle':
            wave = 2 * np.abs(2 * ((t * frequency - self.waveform_phase/(2*np.pi)) % 1) - 1) - 1
        elif self.waveform_type == 'sawtooth':
            wave = 2 * ((t * frequency - self.waveform_phase/(2*np.pi)) % 1) - 1
        else:
            raise ValueError(f"Unsupported waveform type: {self.waveform_type}")

        return wave.astype(np.float32)
    
    def get_audio_segment(self) -> np.ndarray:
       
        song = sf.SoundFile(file=self.song_path, mode='r')
        self.sr = song.samplerate
        total_frames = song.frames

        # Convert start/stop times to frame indices
        start_frame = int(max(0, self.song_start * self.sr))
        if self.song_stop < 0:
            stop_frame = total_frames
        else:
            stop_frame = int(self.song_stop * self.sr)

        # Clamp to valid range
        start_frame = min(start_frame, total_frames)
        stop_frame = min(stop_frame, total_frames)

        # Handle invalid (reversed or zero) ranges
        if stop_frame <= start_frame:
            return np.array([], dtype=np.float32)

        # Calculate how many frames to read
        frames_to_read = stop_frame - start_frame
        self.duration = frames_to_read / self.sr

        # Seek and read safely
        song.seek(start_frame)
        audio_data = song.read(frames_to_read, dtype='float32')

        # Convert to mono if stereo or multichannel
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        return audio_data
    
    def get_onset_signal(self):

        win_samples = max(1, int(self.onset_win * self.sr))
        window = signal.get_window(self.onset_win_type, win_samples)

        energy = np.convolve(self.raw_signal ** 2, window, mode='same')

        self.onset_signal = np.diff(energy, prepend=energy[0])
        self.onset_signal = np.maximum(self.onset_signal, 0)

        self.onset_signal = self.onset_signal - np.mean(self.onset_signal)
        max_val = np.max(np.abs(self.onset_signal))
        if max_val > 0:
            self.onset_signal = self.onset_signal / max_val

        return self.onset_signal
    
    def pll_algorithm(self):

        freq = self.vco_tempo / 60     # Hz
        freq_n = freq / self.sr        # normalized frequency in cycles per sample
        per_n = int(1 / freq_n)        # normalized period in samples

        N = len(self.onset_signal)
        e = np.zeros(N)                # error signal
        Tt = np.ones(N) * per_n        # instantenous period of the vco
        x = np.zeros(N)                # vco output
        vco_phase = 0

        order_0 = np.zeros(N)          # arrays to store
        order_1 = np.zeros(N)          # computations of
        order_2 = np.zeros(N)          # various loop filters

        for t in range(1, N):
            
            vco_phase = (vco_phase + 2 * np.pi / Tt[t-1]) % (2 * np.pi)
            x[t] = np.sin(vco_phase)                               

            start = max(0, t - int(Tt[t-1]))
            if self.loop_filter == 'order_0':
                order_0[t] = x[t] * self.onset_signal[t]
                e[t] = order_0[t]
            elif self.loop_filter == 'order_1':
                order_0[t] = x[t] * self.onset_signal[t]
                order_1[t] = 1 / Tt[t-1] * np.sum(order_0[start:t])
                e[t] = order_1[t]
            elif self.loop_filter == 'order_2':
                order_0[t] = x[t] * self.onset_signal[t]
                order_1[t] = 1 / Tt[t-1] * np.sum(order_0[start:t])
                order_2[t] = 1 / Tt[t-1] * np.sum(order_1[start:t])
                e[t] = order_2[t]
                
            Tt[t] = 1 / max(freq_n + self.loop_gain * e[t], 1e-8)

        self.inst_tempo = 60.0 * self.sr / Tt
        self.vco_out = x

        return self.vco_out, self.inst_tempo

    def generate_metronome(self):

        peaks, _ = signal.find_peaks(self.vco_out, distance=2*self.metro_win)

        clicks = np.zeros_like(self.vco_out)
        clicks[peaks] = 1.0

        click_waveform = np.hanning(self.metro_win)
        metro = np.convolve(clicks, click_waveform, mode='same')

        if self.fix_lag:
            onset_peaks, _ = signal.find_peaks(self.onset_signal, height=0.8)
            lag = []
            for i in range(len(onset_peaks)):
                j = np.argmin(np.abs(peaks - onset_peaks[i]))
                lag.append(onset_peaks[i] - peaks[j])
            lag = np.mean(np.array(lag))

            metro = np.roll(metro, lag)
            if lag > 0:
                metro[:int(lag)] = 0
            elif lag < 0:
                metro[int(lag):] = 0
            
        if np.max(np.abs(metro)) > 0:
            metro /= np.max(np.abs(metro))

        self.metro_signal = metro
        self.metro_lag = lag

        return metro, lag

    def write_audio_output(self, folder_path):
        
    # Default to current directory if none specified
        if folder_path is None or folder_path.strip() == '':
            folder_path = Path(__file__).resolve().parent

        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        output_path = folder / "audio_with_metronome.wav"

        io.wavfile.write(
            output_path,
            self.sr,
            ((0.8 * self.metro_signal + 0.2 * self.raw_signal)).astype(np.float32)
        )

        return output_path

    def process(self):
        self.get_data()
        self.get_onset_signal()
        self.pll_algorithm()
        return self.generate_metronome()