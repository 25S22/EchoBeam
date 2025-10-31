import numpy as np
import sounddevice as sd
from scipy.fft import rfft
import logging
import time
import argparse
import requests

DETECTOR_CONFIG = {
    "chunk_size": 8192,
    "sample_rate": 44100,
    "random_seed": 42,
    "pn_sequence_length": 256,
    
    # Must match watermarker.py!
    "preamble_freq_range": (17500.0, 17800.0),
    "bit_freq_ranges": [
        (17800.0, 18000.0), (18000.0, 18200.0), (18200.0, 18400.0),
        (18400.0, 18600.0), (18600.0, 18800.0), (18800.0, 19000.0),
    ],
    
    "detection_threshold": 0.3,    # Correlation threshold (higher = fewer false positives)
    "cooldown_period": 5,           # Seconds between detections
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioDetector:
    def __init__(self, config, server_url):
        self.config = config
        self.server_url = server_url
        self.last_detection_time = 0
        
        # Generate same PN sequences as watermarker
        self.base_pn_sequences = self._generate_base_sequences()
        
        # Prepare FFT bins
        chunk_size = config['chunk_size']
        sample_rate = config['sample_rate']
        self.fft_freqs = np.fft.rfftfreq(chunk_size, 1 / sample_rate)
        
        # Calculate frequency bin indices
        preamble_range = config['preamble_freq_range']
        self.preamble_indices = (
            np.argmin(np.abs(self.fft_freqs - preamble_range[0])),
            np.argmin(np.abs(self.fft_freqs - preamble_range[1]))
        )
        
        self.bit_indices = []
        for freq_range in config['bit_freq_ranges']:
            self.bit_indices.append((
                np.argmin(np.abs(self.fft_freqs - freq_range[0])),
                np.argmin(np.abs(self.fft_freqs - freq_range[1]))
            ))
        
        # Scale PN sequences to match bin counts
        self.scaled_pn_sequences = {}
        self._scale_pn_sequences()
        
        logging.info(f"AudioDetector ready - listening for {len(config['bit_freq_ranges'])}-bit codes")

    def _generate_pn_sequence(self, seed, length):
        rng = np.random.RandomState(seed)
        sequence = rng.randn(length)
        return sequence / np.linalg.norm(sequence)

    def _generate_base_sequences(self):
        seed = self.config['random_seed']
        length = self.config['pn_sequence_length']
        
        sequences = {'preamble': self._generate_pn_sequence(seed, length)}
        
        bit_sequences = []
        for i in range(len(self.config['bit_freq_ranges'])):
            seq_0 = self._generate_pn_sequence(seed + (i * 2) + 1, length)
            seq_1 = self._generate_pn_sequence(seed + (i * 2) + 2, length)
            bit_sequences.append({'0': seq_0, '1': seq_1})
        
        sequences['bits'] = bit_sequences
        return sequences

    def _scale_pn_sequence(self, base_sequence, num_bins):
        if num_bins <= 0:
            return np.array([])
        base_len = len(base_sequence)
        return np.interp(
            np.linspace(0, base_len - 1, num_bins),
            np.arange(base_len),
            base_sequence
        )

    def _scale_pn_sequences(self):
        start, end = self.preamble_indices
        num_bins = end - start
        self.scaled_pn_sequences['preamble'] = self._scale_pn_sequence(
            self.base_pn_sequences['preamble'], num_bins
        )
        
        self.scaled_pn_sequences['bits'] = []
        for i, (start, end) in enumerate(self.bit_indices):
            num_bins = end - start
            self.scaled_pn_sequences['bits'].append({
                '0': self._scale_pn_sequence(self.base_pn_sequences['bits'][i]['0'], num_bins),
                '1': self._scale_pn_sequence(self.base_pn_sequences['bits'][i]['1'], num_bins)
            })

    def _detect_watermark(self, chunk):
        """Detect watermark in audio chunk using correlation."""
        fft_data = rfft(chunk)
        
        # Check if on cooldown
        if time.time() < self.last_detection_time + self.config['cooldown_period']:
            return None
        
        # 1. Check preamble
        start, end = self.preamble_indices
        if end <= start:
            return None
        
        preamble_signal = fft_data[start:end]
        preamble_pattern = self.scaled_pn_sequences['preamble']
        
        if len(preamble_signal) == 0:
            return None
        
        # Normalize and correlate
        preamble_signal_norm = preamble_signal / (np.linalg.norm(preamble_signal) + 1e-10)
        preamble_pattern_norm = preamble_pattern / (np.linalg.norm(preamble_pattern) + 1e-10)
        preamble_corr = np.abs(np.dot(preamble_signal_norm, preamble_pattern_norm))
        
        # No preamble = no watermark
        if preamble_corr < self.config['detection_threshold']:
            return None
        
        logging.info(f"🔍 Preamble detected! Correlation: {preamble_corr:.3f}")
        
        # 2. Decode bits
        detected_bits = []
        
        for i in range(len(self.bit_indices)):
            start, end = self.bit_indices[i]
            if end <= start:
                detected_bits.append('0')
                continue
            
            bit_signal = fft_data[start:end]
            pattern_0 = self.scaled_pn_sequences['bits'][i]['0']
            pattern_1 = self.scaled_pn_sequences['bits'][i]['1']
            
            if len(bit_signal) == 0:
                detected_bits.append('0')
                continue
            
            # Correlate with both patterns
            bit_signal_norm = bit_signal / (np.linalg.norm(bit_signal) + 1e-10)
            pattern_0_norm = pattern_0 / (np.linalg.norm(pattern_0) + 1e-10)
            pattern_1_norm = pattern_1 / (np.linalg.norm(pattern_1) + 1e-10)
            
            corr_0 = np.abs(np.dot(bit_signal_norm, pattern_0_norm))
            corr_1 = np.abs(np.dot(bit_signal_norm, pattern_1_norm))
            
            detected_bits.append('1' if corr_1 > corr_0 else '0')
        
        detected_code = ''.join(detected_bits)
        
        logging.info(f"🎵 WATERMARK DETECTED: '{detected_code}' (confidence: {preamble_corr:.3f})")
        self.last_detection_time = time.time()
        
        return detected_code

    def _trigger_server(self, detected_code):
        """Send detection event to server."""
        try:
            logging.info(f"📡 Sending code '{detected_code}' to server...")
            response = requests.post(
                self.server_url,
                json={"code": detected_code},
                timeout=3
            )
            
            if response.ok:
                data = response.json()
                action = data.get('action', 'UNKNOWN')
                logging.info(f"✓ Server response: {action}")
            else:
                logging.warning(f"Server error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Cannot reach server at {self.server_url}: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            logging.warning(f"Audio status: {status}")
        
        try:
            chunk = indata.flatten()
            detected_code = self._detect_watermark(chunk)
            
            if detected_code:
                self._trigger_server(detected_code)
                
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    def start_listening(self):
        """Start microphone stream."""
        logging.info(f"🎤 Starting microphone (sample rate: {self.config['sample_rate']}Hz)")
        logging.info(f"🔍 Listening for watermarks in {self.config['preamble_freq_range'][0]:.0f}-{max([r[1] for r in self.config['bit_freq_ranges']]):.0f}Hz")
        
        with sd.InputStream(
            channels=1,
            samplerate=self.config['sample_rate'],
            blocksize=self.config['chunk_size'],
            callback=self.audio_callback,
            dtype='float32'
        ):
            print("\n" + "="*60)
            print("✅ DETECTOR RUNNING - Press Ctrl+C to stop")
            print(f"   Server: {self.server_url}")
            print(f"   Cooldown: {self.config['cooldown_period']}s between detections")
            print("="*60 + "\n")
            
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\n🛑 Detector stopped by user")

def main():
    parser = argparse.ArgumentParser(description="Real-time audio watermark detector")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:5000/trigger",
        help="Server URL for sending detections"
    )
    args = parser.parse_args()
    
    try:
        detector = AudioDetector(config=DETECTOR_CONFIG, server_url=args.server)
        detector.start_listening()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
