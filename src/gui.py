import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import soundfile as sf
from processing import BeatTrackProcessor
import os
import numpy as np
import math
import threading
import pathlib
import matplotlib.pyplot as plt


class BeatTrackGUI:
    def __init__(self, root):
        
        self.processor = BeatTrackProcessor()

        self.root = root
        self.root.title("PLL Beat Tracker")
        # self.root.geometry("1000x800")

        self.create_input_selection()
        self.create_song_options()
        self.create_waveform_options()
        self.create_pll_options()
        self.create_plot_options()
        self.create_audio_options()

        self.song_options_frame.grid_columnconfigure(0, minsize=100)
        self.song_options_frame.grid_columnconfigure(1, minsize=150)
        self.song_options_frame.grid_columnconfigure(2, minsize=75)
        self.waveform_options_frame.grid_columnconfigure(0, minsize=100)
        self.waveform_options_frame.grid_columnconfigure(1, minsize=150)
        self.waveform_options_frame.grid_columnconfigure(2, minsize=75)

        self.update_input_mode()
        self.set_default_audio_folder()

    # --- Layout Functions --- #
    def create_input_selection(self):

        self.input_mode = tk.StringVar(value='song')
        
        self.input_selection_frame = ttk.LabelFrame(self.root, text='Input Selection', padding=5)
        self.input_selection_frame.grid(row=0, column=0, sticky='nswe', padx=5, pady= 5)

        ttk.Radiobutton(self.input_selection_frame,
                        text="Song", variable=self.input_mode, value="song",
                        command=self.update_input_mode
                        ).grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        ttk.Radiobutton(self.input_selection_frame,
                        text="Waveform", variable=self.input_mode, value="waveform",
                        command=self.update_input_mode
                        ).grid(row=0, column=1, sticky='ew', padx=5, pady=5)  

    def create_song_options(self):  
        
        self.song_path = tk.StringVar()
        self.song_label = tk.StringVar(value='No File Loaded...')
        self.section_dur_str = tk.StringVar(value="--:--")
        self.song_dur = 0
        self.section_start = tk.DoubleVar()
        self.section_start_str = tk.StringVar()
        self.section_end = tk.DoubleVar()
        self.section_end_str = tk.StringVar()

        self.section_start.trace_add('write', self.update_section_times)
        self.section_end.trace_add('write', self.update_section_times)

        self.song_options_frame = ttk.LabelFrame(self.root, text='Song Options', padding=5)
        self.song_options_frame.grid(row=1,column=0, rowspan=2, sticky='nswe',padx=5,pady=5)

        ttk.Label(self.song_options_frame, textvariable=self.song_label,
                  width=40, anchor='w'
                  ).grid(row=0, column=0,columnspan=2, sticky='w', padx=5, pady=5)
        ttk.Button(self.song_options_frame, text='Load',command=self.load_song
                   ).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.song_options_frame, text='Duration:'
            ).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(self.song_options_frame, textvariable=self.section_dur_str
            ).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(self.song_options_frame, text="Start:"
                  ).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.section_start_scale = ttk.Scale(self.song_options_frame,
                  from_=0, to=60, variable=self.section_start,
                  orient='horizontal', length=100, state='disabled')
        self.section_start_scale.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.song_options_frame, textvariable=self.section_start_str
                  ).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(self.song_options_frame, text="End:"
                  ).grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.section_end_scale = ttk.Scale(self.song_options_frame,
                  from_=0, to=60, variable=self.section_end,
                  orient='horizontal', length=100, state='disabled')
        self.section_end_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.song_options_frame, textvariable=self.section_end_str
                  ).grid(row=3, column=2, padx=5, pady=5)
    
    def create_waveform_options(self):

        self.waveform_type = tk.StringVar(value='sine')
        self.waveform_tempo = tk.DoubleVar(value=120)
        self.waveform_phase = tk.DoubleVar(value=0)
        self.waveform_duty = tk.DoubleVar(value=0.5)
        self.waveform_dur = tk.DoubleVar(value=10)

        self.waveform_type.trace_add('write', self.update_dc_scale)

        self.waveform_options_frame = ttk.LabelFrame(self.root, text='Waveform Options', padding=5)
        self.waveform_options_frame.grid(row=3,column=0, rowspan=2, sticky='nswe',padx=5,pady=5)

        ttk.Label(self.waveform_options_frame, text='Type:'
                  ).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Combobox(self.waveform_options_frame, textvariable=self.waveform_type,
                     values=('sine','square','triangle', 'sawtooth'), state='readonly'
                     ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.waveform_options_frame, text='Tempo (BPM):'
                  ).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Scale(self.waveform_options_frame,
                  from_=60, to=180, variable=self.waveform_tempo,
                  orient='horizontal', length=100,
                  command=lambda x: self.waveform_tempo.set(round(float(x)))
                  ).grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.waveform_options_frame, textvariable=self.waveform_tempo
                  ).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(self.waveform_options_frame, text='Phase (deg):'
                  ).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Scale(self.waveform_options_frame,
                  from_=0, to=360, variable=self.waveform_phase,
                  orient='horizontal', length=100,
                  command=lambda x: self.waveform_phase.set(round(float(x)))
                  ).grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.waveform_options_frame, textvariable=self.waveform_phase
                  ).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(self.waveform_options_frame, text='Duty Cycle:'
                  ).grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.waveform_duty_scale = ttk.Scale(self.waveform_options_frame,
                  from_=0.01, to=0.99, variable=self.waveform_duty,
                  orient='horizontal', length=100, state='disabled',
                  command=lambda x: self.waveform_duty.set(round(float(x), 2)))
        self.waveform_duty_scale.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.waveform_options_frame, textvariable=self.waveform_duty
                  ).grid(row=3, column=2, padx=5, pady=5)

    def create_pll_options(self):
        
        self.loop_filter = tk.StringVar(value='order_1')
        self.vco_tempo = tk.DoubleVar(value=120)
        self.loop_gain = tk.DoubleVar(value= 0.001)
        self.loop_gain_mag = tk.DoubleVar(value=-3)
        self.fix_lag = tk.BooleanVar(value=True)
        self.progress_var = tk.IntVar(value=0)

        self.pll_options_frame = ttk.LabelFrame(self.root, text='PLL Options', padding=5)
        self.pll_options_frame.grid(row=0,column=1, rowspan=2, sticky='nwe',padx=5,pady=5)

        ttk.Label(self.pll_options_frame, text='Loop Filter:'
                  ).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Combobox(self.pll_options_frame, textvariable=self.loop_filter,
                     values=('order_0','order_1','order_2'), state='readonly'
                     ).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.pll_options_frame, text='Loop Gain:'
                  ).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        ttk.Scale(self.pll_options_frame,
                  from_=-5.0, to=0.0, variable=self.loop_gain_mag,
                  orient='horizontal', length=100,
                  command=lambda x: self.calculate_loop_gain(float(x))
                  ).grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.pll_options_frame, textvariable=self.loop_gain,
                  ).grid(row=1, column=2, padx=5, pady=5)


        ttk.Label(self.pll_options_frame, text='VCO Tempo (BPM):'
                  ).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        ttk.Scale(self.pll_options_frame,
                  from_=60, to=180, variable=self.vco_tempo,
                  orient='horizontal', length=100,
                  command=lambda x: self.vco_tempo.set(round(float(x)))
                  ).grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Label(self.pll_options_frame, textvariable=self.vco_tempo
                  ).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Checkbutton(self.pll_options_frame, text='Compensate Metronome Lag',
                        variable=self.fix_lag
                        ).grid(row=3, column=0,columnspan=2, sticky='w', padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(self.pll_options_frame, orient='horizontal',
                                            length=150, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        ttk.Button(self.pll_options_frame, text='Run!', command=self.run_pll_thread
                   ).grid(row=4, column=2, padx=5, pady=5)

    def create_plot_options(self):

        self.plot_metro = tk.BooleanVar(value=True)
        self.plot_inst_tempo = tk.BooleanVar(value=True)

        self.plot_options_frame = ttk.LabelFrame(self.root, text='Plot Options', padding=5)
        self.plot_options_frame.grid(row=2, column=1, rowspan=2, sticky='nwe', padx=5, pady=5)
        self.plot_options_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Button(self.plot_options_frame, text='Show Metronome Plot',
                   command=self.show_metronome_plot
                   ).grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        
        ttk.Button(self.plot_options_frame, text='Show Tempo Locking Plot',
                   command=self.show_tempo_plot
                   ).grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
    def create_audio_options(self):
        
        self.audio_folder_path = tk.StringVar(value='')

        self.audio_options_frame = ttk.LabelFrame(self.root, text='Audio Options', padding=5)
        self.audio_options_frame.grid(row=4, column=1, sticky='nwe', padx=5, pady=5)

        ttk.Label(self.audio_options_frame, text='Save Folder:'
                  ).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        ttk.Label(self.audio_options_frame, textvariable=self.audio_folder_path,
                  width=30, anchor='w'
                  ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.audio_options_frame, text='Browse',
                   command=self.choose_audio_folder
                   ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Button(self.audio_options_frame, text='Save Audio Output',
                   command=self.save_audio_output
                   ).grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky='ew')


    # --- Helper Functions --- #
    def update_input_mode(self):
        mode = self.input_mode.get()
        if mode == 'song':
            self.enable_frame(self.song_options_frame)
            self.disable_frame(self.waveform_options_frame)
        elif mode == 'waveform':
            self.disable_frame(self.song_options_frame)
            self.enable_frame(self.waveform_options_frame)
        
        if self.song_label.get() == 'No File Loaded...':
            self.section_start_scale.state(['disabled'])
            self.section_end_scale.state(['disabled'])

        if self.waveform_type.get() != 'square':
            self.waveform_duty_scale.state(['disabled'])

    def disable_frame(self, frame):
        """Disable all widgets inside a frame"""
        for child in frame.winfo_children():
            try:
                child.configure(state='disabled')
            except tk.TclError:
                pass

    def enable_frame(self, frame):
        """Enable all widgets inside a frame."""
        for child in frame.winfo_children():
            try:
                child.configure(state='normal')
            except tk.TclError:
                pass
    
    def disable_ui(self):
        for frame in [self.input_selection_frame,
                      self.song_options_frame,
                      self.waveform_options_frame,
                      self.pll_options_frame]:
            self.disable_frame(frame)
    
    def enable_ui(self):
        for frame in [self.input_selection_frame,
                      self.song_options_frame,
                      self.waveform_options_frame,
                      self.pll_options_frame]:
            self.enable_frame(frame)

    def load_song(self):
        
        song_path = filedialog.askopenfilename(
            title='Select Audio File',
            filetypes=[
                ("WAV files", "*.wav"),
                ("FLAC files", "*.flac"),
                ("OGG files", "*.ogg"),
                ("All supported audio files", "*.wav *.flac *.ogg")
                ]
            )
        
        if not song_path:
            return
        
        try:
            with sf.SoundFile(song_path) as f:
                self.song_path.set(song_path)
                self.song_label.set(os.path.basename(song_path))
                self.section_start.set(0)
                self.song_dur = f.frames / f.samplerate
            
                self.section_start_scale.state(['!disabled'])
                self.section_end_scale.state(['!disabled'])
                self.section_start_scale.configure(to=self.song_dur)
                self.section_end_scale.configure(to=self.song_dur)

                self.section_end.set(self.song_dur)


        except RuntimeError as e:
            messagebox.showerror("File Error", f"Cannot open file:\n{e}")

    def update_section_times(self, *args):
        start = self.section_start.get()
        end = self.section_end.get()

        # Enforce logical order
        if end < start:
            self.section_end.set(start)
            end = start

        # Dynamically adjust slider ranges
        self.section_end_scale.configure(from_=start)
        self.section_start_scale.configure(to=end)

        # Update display labels
        dur = end - start
        self.section_start_str.set(self.seconds_to_mmss(int(start)))
        self.section_end_str.set(self.seconds_to_mmss(int(end)))
        self.section_dur_str.set(self.seconds_to_mmss(int(dur)))

    def seconds_to_mmss(self, seconds: int) -> str:
        """Convert integer seconds to MM:SS formatted string."""
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    def update_dc_scale(self, *args):
        if self.waveform_type.get() == 'square':
            self.waveform_duty_scale.state(['!disabled'])
        else:
            self.waveform_duty_scale.state(['disabled'])

    def calculate_loop_gain(self, mag):
       
       frac, whole = math.modf(mag) 
       targets = [0, -0.30103, -0.69897, -1]
       frac = min(targets, key=lambda t: abs(frac - t))
       quantized = whole + frac
       gain = 10 ** quantized

       self.loop_gain_mag.set(quantized)
       self.loop_gain.set(round(gain, 5))

    def run_pll_thread(self):
        threading.Thread(target=self.run_pll, daemon=True).start()

    def run_pll(self):

        self.disable_ui()
        self.progress_var.set(0)
        self.root.update_idletasks()
        
        try:
            # Step 0: update processor parameters
            if self.input_mode.get() == 'song':
                self.processor.update_params(
                    signal_type=self.input_mode.get(),

                    song_path= self.song_path.get(),
                    song_start=self.section_start.get(),
                    song_stop=self.section_end.get(),

                    loop_gain= self.loop_gain.get(),
                    vco_tempo= self.vco_tempo.get(),
                    loop_filter= self.loop_filter.get(),
                    fix_lag= self.fix_lag.get()
                )
            else:
                self.processor.update_params(
                    signal_type=self.input_mode.get(),

                    waveform_type=self.waveform_type.get(),
                    waveform_tempo= self.waveform_tempo.get(),
                    waveform_duty= self.waveform_duty.get(),
                    waveform_phase= self.waveform_phase.get(),

                    loop_gain= self.loop_gain.get(),
                    vco_tempo= self.vco_tempo.get(),
                    loop_filter= self.loop_filter.get(),
                    fix_lag= self.fix_lag.get()
                )

            # Step 1: Load / generate data
            self.processor.get_data()
            self.progress_var.set(10)
            self.root.update_idletasks()

            # Step 2: Compute onset signal
            self.processor.get_onset_signal()
            self.progress_var.set(30)
            self.root.update_idletasks()

            # Step 3: Run PLL algorithm
            self.processor.pll_algorithm()
            self.progress_var.set(80)
            self.root.update_idletasks()

            # Step 4: Generate metronome
            metro, _ = self.processor.generate_metronome()
            self.progress_var.set(100)
            self.root.update_idletasks()

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

        finally:
            self.enable_ui()

    def set_default_audio_folder(self):
        default_folder = pathlib.Path(__file__).parent.parent / 'audio'
        default_folder.mkdir(exist_ok=True)
        self.audio_folder_path.set(str(default_folder))

    def choose_audio_folder(self):
        folder = filedialog.askdirectory(initialdir=self.audio_folder_path.get(),
                                         title="Select Folder to Save Audio")
        if folder:
            self.audio_folder_path.set(folder)

    def save_audio_output(self):
        folder = self.audio_folder_path.get()
        try:
            output_path = self.processor.write_audio_output(folder)
            messagebox.showinfo("Saved", f"Audio output saved to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save audio:\n{e}")

    def show_metronome_plot(self):
        onset = self.processor.onset_signal
        metro = self.processor.metro_signal
        sr = self.processor.sr

        t = np.arange(len(onset)) / sr  # convert sample index to seconds

        plt.figure(figsize=(10, 4))
        plt.plot(t, onset, label='Onset Signal')
        plt.plot(t, metro, label='Metronome', color='red', alpha=0.7)
        plt.title("Metronome vs Onset Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_tempo_plot(self):
        inst_tempo = self.processor.inst_tempo
        avg_tempo = np.mean(inst_tempo)
        sr = self.processor.sr

        t = np.arange(len(inst_tempo)) / sr

        plt.figure(figsize=(10, 4))
        plt.plot(inst_tempo, label='Instantaneous Tempo', color='green')
        plt.axhline(y=avg_tempo, color='red', linestyle='--', label=f'Average Tempo: {avg_tempo:.2f} BPM')
        plt.title("Instantaneous Tempo")
        plt.xlabel("Time (s)")
        plt.ylabel("BPM")
        plt.legend()
        plt.tight_layout()
        plt.show()

root = tk.Tk()
BeatTrackGUI(root)
root.mainloop()