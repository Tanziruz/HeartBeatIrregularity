"""
Make predictions using TensorFlow Lite model
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import librosa
import h5py

from utils.audio_feature_extractor import LogMelExtractor
from utils.util import read_audio

EPS = 1E-8


def standard_normal_variate(data):
    """Normalize data using standard normal variate"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


class TFLitePredictor:
    def __init__(self, tflite_model_path, config=None):
        """
        Initialize TFLite predictor
        
        Args:
            tflite_model_path: Path to .tflite model file
            config: Configuration dict for preprocessing
        """
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Default config
        if config is None:
            config = {
                'in_channel': 3,
                'duration': 5,
                'delta': True,
                'norm': True,
                'mel_bins': 128
            }
        
        self.config = config
        self.class_labels = ['Normal', 'Abnormal']
        
        print(f"TFLite model loaded: {tflite_model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        # Read audio with filtering
        audio, sr = read_audio(audio_path, filter=True)
        
        # Extract mel-spectrogram
        feature = LogMelExtractor(audio, sr, mel_bins=self.config['mel_bins'], 
                                 log=True, snv=False)
        
        # Apply normalization
        if self.config['norm']:
            feature = standard_normal_variate(feature)
        
        # Add deltas
        if self.config['delta']:
            delta = librosa.feature.delta(feature)
            delta_2 = librosa.feature.delta(delta)
            feature = np.concatenate((feature, delta, delta_2), axis=0)
        
        # Pad or crop to fixed duration
        cycle_len = int(self.config['duration'] * 1000 / 15)
        mel_bins, num_frames = feature.shape
        
        if num_frames >= cycle_len:
            start_ind = int((num_frames - cycle_len) / 2)
            feature_pad = feature[:, start_ind:start_ind + cycle_len]
        elif num_frames < cycle_len:
            feature_pad = np.pad(feature, ((0, 0), (0, cycle_len - num_frames)), 
                                mode='wrap')
        else:
            feature_pad = feature
        
        return feature_pad
    
    def predict_file(self, audio_path):
        """
        Make prediction on a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        feature = self.extract_features(audio_path)
        
        # Prepare input (add batch dimension)
        input_data = np.expand_dims(feature, axis=0).astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output (log probabilities)
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Convert to probabilities
        probabilities = np.exp(output[0])
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        return {
            'file': Path(audio_path).name,
            'predicted_class': int(predicted_class),
            'predicted_label': self.class_labels[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'Normal': float(probabilities[0]),
                'Abnormal': float(probabilities[1])
            }
        }
    
    def predict_batch(self, audio_paths):
        """Make predictions on multiple files"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_file(audio_path)
                results.append(result)
                print(f"✓ {result['file']}: {result['predicted_label']} "
                      f"(confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {audio_path}: {str(e)}")
                results.append({
                    'file': Path(audio_path).name,
                    'error': str(e)
                })
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict using TFLite model')
    parser.add_argument('--model', type=str, default='model.tflite',
                        help='Path to TFLite model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to audio file or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')
    parser.add_argument('--in_channel', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--duration', type=int, default=5,
                        help='Audio duration in seconds')
    parser.add_argument('--no-delta', action='store_true',
                        help='Disable delta features')
    parser.add_argument('--no-norm', action='store_true',
                        help='Disable normalization')
    
    args = parser.parse_args()
    
    # Config
    config = {
        'in_channel': args.in_channel,
        'duration': args.duration,
        'delta': not args.no_delta,
        'norm': not args.no_norm,
        'mel_bins': 128
    }
    
    # Initialize predictor
    predictor = TFLitePredictor(args.model, config)
    
    # Get input files
    input_path = Path(args.input)
    
    if input_path.is_dir():
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac']:
            audio_files.extend(input_path.glob(f'*{ext}'))
        print(f"\nFound {len(audio_files)} audio files")
        results = predictor.predict_batch([str(f) for f in audio_files])
        
    elif input_path.is_file():
        print(f"\nPredicting on: {input_path}")
        result = predictor.predict_file(str(input_path))
        results = [result]
        print(f"\nPrediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(f"  Normal: {result['probabilities']['Normal']:.2%}")
        print(f"  Abnormal: {result['probabilities']['Abnormal']:.2%}")
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    # Save results
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    if successful:
        normal = sum(1 for r in successful if r['predicted_class'] == 0)
        abnormal = sum(1 for r in successful if r['predicted_class'] == 1)
        print(f"\nSummary:")
        print(f"  Total: {len(successful)}")
        print(f"  Normal: {normal} ({normal/len(successful):.1%})")
        print(f"  Abnormal: {abnormal} ({abnormal/len(successful):.1%})")


if __name__ == '__main__':
    main()
