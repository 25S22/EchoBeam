import numpy as np
import logging
from scipy.io import wavfile
from scipy.fft import rfft, irfft
import argparse

WATERMARK_CONFIG = {
    "chunk_size": 8192,
    "watermark_strength": 0.7,      # Stronger for air transmission
    "random_seed": 42,
    "pn_sequence_length": 256,
    
    # High-frequency bands (inaudible but survives speaker->mic)
    "preamble_freq_range": (17500.0, 17800.0),
    "bit_freq_ranges": [
        (17800.0, 18000.0), (18000.0, 18200.0), (18200.0, 18400.0),
        (18400.0, 18600.0), (18600.0, 18800.0), (18800.0, 19000.0),
    ],
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioWatermarker:
    def __init__(self, config=None):
        self.config = WATERMARK_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.base_pn_sequences = self._generate_base_sequences()
        self.fft_freqs = None
        self.preamble_indices = None
        self.bit_indices = None
        self.scaled_pn_sequences = {}
        
        logging.info(f"Watermarker ready - {len(self.config['bit_freq_ranges'])} bit capacity")

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

    def _prepare_fft_data(self, sample_rate, n_samples):
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

    def _embed_in_chunk(self, chunk, secret_code):
        n_samples = len(chunk)
        fft_data = rfft(chunk)
        
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

    def embed(self, input_path, output_path, secret_code):
        num_bits = len(self.config['bit_freq_ranges'])
        if not all(c in '01' for c in secret_code) or len(secret_code) != num_bits:
            raise ValueError(f"Code must be a {num_bits}-bit binary string.")
        
        sample_rate, audio_data = wavfile.read(input_path)
        
        max_freq = max([r[1] for r in self.config['bit_freq_ranges']])
        if sample_rate / 2 <= max_freq:
            raise ValueError(f"Sample rate {sample_rate}Hz too low. Need >{max_freq*2}Hz (try 44100Hz)")
        
        original_dtype = audio_data.dtype
        if audio_data.ndim > 1:
            logging.info("Converting stereo to mono")
            audio_data = audio_data.mean(axis=1)
        
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            audio_data = audio_data.astype(np.float64) / max(np.abs(info.min), info.max)

        chunk_size = self.config["chunk_size"]
        self._prepare_fft_data(sample_rate, chunk_size)
        
        watermarked_audio = np.array([], dtype=np.float64)
        logging.info(f"Embedding code '{secret_code}' into {input_path}")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            watermarked_chunk = self._embed_in_chunk(chunk, secret_code)
            watermarked_audio = np.append(watermarked_audio, watermarked_chunk)

        final_audio = watermarked_audio[:len(audio_data)]
        
        if np.issubdtype(original_dtype, np.integer):
            final_audio = np.clip(final_audio, -1.0, 1.0)
            if '16' in str(original_dtype):
                final_audio = (final_audio * 32767).astype(original_dtype)
            elif '32' in str(original_dtype):
                final_audio = (final_audio * 2147483647).astype(original_dtype)

        wavfile.write(output_path, sample_rate, final_audio)
        logging.info(f"✓ Watermarked audio saved: {output_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Embed audio watermark")
    parser.add_argument('command', choices=['embed'], help='Command')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('code', help='6-bit binary code (e.g., 101101)')
    args = parser.parse_args()
    
    try:
        AudioWatermarker().embed(args.input, args.output, args.code)
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
    
