import numpy as np
import logging
from scipy.io import wavfile
from scipy.fft import rfft, irfft
from scipy.signal import correlate
import argparse
import os

DEFAULT_CONFIG = {
    "chunk_size": 8192,           # Larger chunks for better frequency resolution
    "watermark_strength": 0.3,    # More conservative strength
    "random_seed": 42,            # Secret key for PN sequences
    "pn_sequence_length": 256,    # Longer sequences for better detection
    
    # Frequency ranges (high-frequency, typically inaudible)
    "preamble_freq_range": (17500.0, 17800.0),
    "bit_freq_ranges": [
        (17800.0, 18000.0), (18000.0, 18200.0), (18200.0, 18400.0),
        (18400.0, 18600.0), (18600.0, 18800.0), (18800.0, 19000.0),
        (19000.0, 19200.0), (19200.0, 19400.0),  # 8 bits total
    ],
    
    # Detection parameters
    "detection_threshold": 0.15,   # Correlation threshold for detection
    "min_detections": 3,           # Minimum detections to confirm watermark
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioWatermarker:
    def __init__(self, config=None):
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.base_pn_sequences = self._generate_base_sequences()
        self.fft_freqs = None
        self.preamble_indices = None
        self.bit_indices = None
        self.scaled_pn_sequences = {}
        
        logging.info(f"AudioWatermarker initialized - {len(self.config['bit_freq_ranges'])} bit capacity")

    def _generate_pn_sequence(self, seed, length):
        """Generate normalized pseudo-random sequence."""
        rng = np.random.RandomState(seed)
        sequence = rng.randn(length)
        return sequence / np.linalg.norm(sequence)

    def _generate_base_sequences(self):
        """Generate all PN sequences from master seed."""
        seed = self.config['random_seed']
        length = self.config['pn_sequence_length']
        
        sequences = {
            'preamble': self._generate_pn_sequence(seed, length)
        }
        
        bit_sequences = []
        for i in range(len(self.config['bit_freq_ranges'])):
            seq_0 = self._generate_pn_sequence(seed + (i * 2) + 1, length)
            seq_1 = self._generate_pn_sequence(seed + (i * 2) + 2, length)
            bit_sequences.append({'0': seq_0, '1': seq_1})
        
        sequences['bits'] = bit_sequences
        return sequences

    def _prepare_fft_data(self, sample_rate, n_samples):
        """Calculate FFT frequencies and indices."""
        self.fft_freqs = np.fft.rfftfreq(n_samples, 1 / sample_rate)
        
        preamble_range = self.config['preamble_freq_range']
        self.preamble_indices = (
            np.argmin(np.abs(self.fft_freqs - preamble_range[0])),
            np.argmin(np.abs(self.fft_freqs - preamble_range[1]))
        )
        
        self.bit_indices = []
        for freq_range in self.config['bit_freq_ranges']:
            self.bit_indices.append((
                np.argmin(np.abs(self.fft_freqs - freq_range[0])),
                np.argmin(np.abs(self.fft_freqs - freq_range[1]))
            ))
        
        self._scale_pn_sequences()

    def _scale_pn_sequence(self, base_sequence, num_bins):
        """Scale PN sequence to fit FFT bins."""
        if num_bins <= 0:
            return np.array([])
        base_len = len(base_sequence)
        return np.interp(
            np.linspace(0, base_len - 1, num_bins),
            np.arange(base_len),
            base_sequence
        )

    def _scale_pn_sequences(self):
        """Pre-scale all PN sequences."""
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

    def _embed_in_chunk(self, chunk, secret_code):
        """Embed watermark in a single chunk using DSSS."""
        n_samples = len(chunk)
        fft_data = rfft(chunk)
        
        # Calculate adaptive strength based on local signal energy
        all_indices = np.concatenate(
            [np.arange(self.preamble_indices[0], self.preamble_indices[1])] +
            [np.arange(idx[0], idx[1]) for idx in self.bit_indices if idx[1] > idx[0]]
        )
        
        if len(all_indices) > 0:
            avg_mag = np.mean(np.abs(fft_data[all_indices]))
            if avg_mag < 1e-6:
                avg_mag = 1.0
        else:
            avg_mag = 1.0
                
        strength = avg_mag * self.config['watermark_strength']

        # Embed preamble
        start, end = self.preamble_indices
        if end > start:
            fft_data[start:end] += self.scaled_pn_sequences['preamble'] * strength

        # Embed bits
        for i, bit in enumerate(secret_code):
            start, end = self.bit_indices[i]
            if end > start:
                pn_to_embed = self.scaled_pn_sequences['bits'][i][bit]
                fft_data[start:end] += pn_to_embed * strength

        return irfft(fft_data, n=n_samples)

    def _detect_in_chunk(self, chunk):
        """Detect watermark in a chunk using correlation."""
        n_samples = len(chunk)
        fft_data = rfft(chunk)
        
        # Check preamble first
        start, end = self.preamble_indices
        if end <= start:
            return None, []
        
        preamble_signal = fft_data[start:end]
        preamble_pattern = self.scaled_pn_sequences['preamble']
        
        if len(preamble_signal) == 0 or len(preamble_pattern) == 0:
            return None, []
        
        # Normalize and correlate
        preamble_signal_norm = preamble_signal / (np.linalg.norm(preamble_signal) + 1e-10)
        preamble_pattern_norm = preamble_pattern / (np.linalg.norm(preamble_pattern) + 1e-10)
        preamble_corr = np.abs(np.dot(preamble_signal_norm, preamble_pattern_norm))
        
        # If preamble not detected, no watermark
        threshold = self.config['detection_threshold']
        if preamble_corr < threshold:
            return None, []
        
        # Decode bits
        detected_bits = []
        bit_confidences = []
        
        for i in range(len(self.bit_indices)):
            start, end = self.bit_indices[i]
            if end <= start:
                detected_bits.append('?')
                bit_confidences.append(0.0)
                continue
            
            bit_signal = fft_data[start:end]
            pattern_0 = self.scaled_pn_sequences['bits'][i]['0']
            pattern_1 = self.scaled_pn_sequences['bits'][i]['1']
            
            if len(bit_signal) == 0:
                detected_bits.append('?')
                bit_confidences.append(0.0)
                continue
            
            # Correlate with both patterns
            bit_signal_norm = bit_signal / (np.linalg.norm(bit_signal) + 1e-10)
            pattern_0_norm = pattern_0 / (np.linalg.norm(pattern_0) + 1e-10)
            pattern_1_norm = pattern_1 / (np.linalg.norm(pattern_1) + 1e-10)
            
            corr_0 = np.abs(np.dot(bit_signal_norm, pattern_0_norm))
            corr_1 = np.abs(np.dot(bit_signal_norm, pattern_1_norm))
            
            # Choose stronger correlation
            if corr_0 > corr_1:
                detected_bits.append('0')
                bit_confidences.append(corr_0)
            else:
                detected_bits.append('1')
                bit_confidences.append(corr_1)
        
        return ''.join(detected_bits), bit_confidences

    def embed(self, input_path, output_path, secret_code):
        """Embed watermark into audio file."""
        num_bits = len(self.config['bit_freq_ranges'])
        if not all(c in '01' for c in secret_code) or len(secret_code) != num_bits:
            raise ValueError(f"Code must be a {num_bits}-bit binary string (e.g., '10110101').")
        
        sample_rate, audio_data = wavfile.read(input_path)
        
        # Check sample rate
        max_freq = max([r[1] for r in self.config['bit_freq_ranges']])
        nyquist = sample_rate / 2
        if max_freq >= nyquist:
            logging.warning(f"Sample rate {sample_rate}Hz too low for frequency range (need >{max_freq*2}Hz)")
            logging.warning(f"Watermark may not work properly!")
        
        original_dtype = audio_data.dtype
        if audio_data.ndim > 1:
            logging.info("Converting stereo to mono")
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            audio_data = audio_data.astype(np.float64) / max(np.abs(info.min), info.max)
        elif not np.issubdtype(original_dtype, np.floating):
            raise ValueError(f"Unsupported audio dtype: {original_dtype}")

        chunk_size = self.config["chunk_size"]
        self._prepare_fft_data(sample_rate, chunk_size)
        
        watermarked_audio = np.array([], dtype=np.float64)
        logging.info(f"Embedding {num_bits}-bit code: '{secret_code}'")
        
        num_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            watermarked_chunk = self._embed_in_chunk(chunk, secret_code)
            watermarked_audio = np.append(watermarked_audio, watermarked_chunk)

        final_audio = watermarked_audio[:len(audio_data)]
        
        # Convert back to original format
        if np.issubdtype(original_dtype, np.integer):
            final_audio = np.clip(final_audio, -1.0, 1.0)
            if '16' in str(original_dtype):
                final_audio = (final_audio * 32767).astype(original_dtype)
            elif '32' in str(original_dtype):
                final_audio = (final_audio * 2147483647).astype(original_dtype)
        else:
            final_audio = final_audio.astype(original_dtype)

        wavfile.write(output_path, sample_rate, final_audio)
        logging.info(f"✓ Watermarked audio saved: {output_path}")
        return True

    def detect(self, input_path):
        """Detect and decode watermark from audio file."""
        sample_rate, audio_data = wavfile.read(input_path)
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Convert to float
        original_dtype = audio_data.dtype
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            audio_data = audio_data.astype(np.float64) / max(np.abs(info.min), info.max)

        chunk_size = self.config["chunk_size"]
        self._prepare_fft_data(sample_rate, chunk_size)
        
        detections = []
        logging.info("Scanning for watermark...")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            detected_code, confidences = self._detect_in_chunk(chunk)
            if detected_code:
                detections.append((detected_code, confidences))
        
        if len(detections) < self.config['min_detections']:
            logging.warning(f"⚠ No watermark detected ({len(detections)} detections, need {self.config['min_detections']})")
            return None
        
        # Vote on most common code
        from collections import Counter
        codes = [d[0] for d in detections]
        most_common = Counter(codes).most_common(1)[0]
        detected_code = most_common[0]
        count = most_common[1]
        
        # Calculate average confidence
        avg_confidence = np.mean([np.mean(d[1]) for d in detections if d[0] == detected_code])
        
        logging.info(f"✓ Watermark detected: '{detected_code}'")
        logging.info(f"  Confidence: {avg_confidence:.3f}, Detections: {count}/{len(detections)} chunks")
        
        return {
            'code': detected_code,
            'confidence': avg_confidence,
            'detection_rate': count / len(detections),
            'total_detections': count
        }


def main():
    parser = argparse.ArgumentParser(
        description="Audio Watermarking with DSSS - Embed and detect secret codes in audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Embed:   python watermarker.py embed input.wav output.wav 10110101
  Detect:  python watermarker.py detect watermarked.wav
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed watermark into audio')
    embed_parser.add_argument('input', help='Input WAV file')
    embed_parser.add_argument('output', help='Output WAV file')
    embed_parser.add_argument('code', help=f'Secret binary code ({len(DEFAULT_CONFIG["bit_freq_ranges"])} bits)')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark from audio')
    detect_parser.add_argument('input', help='Input WAV file to check')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        watermarker = AudioWatermarker()
        
        if args.command == 'embed':
            watermarker.embed(args.input, args.output, args.code)
        elif args.command == 'detect':
            result = watermarker.detect(args.input)
            if result:
                print(f"\n{'='*50}")
                print(f"Detected Code: {result['code']}")
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Detection Rate: {result['detection_rate']:.1%}")
                print(f"{'='*50}\n")
            else:
                print("\nNo watermark found in this audio file.\n")
                
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
