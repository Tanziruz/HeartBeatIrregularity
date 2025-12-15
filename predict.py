import argparse
import torch
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import librosa
import soundfile

from model.model import VGG_11
from utils.audio_feature_extractor import LogMelExtractor
from utils.util import read_audio

EPS = 1E-8

def standard_normal_variate(data):
    """Normalize data using standard normal variate"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


class HeartSoundPredictor:
    def __init__(self, checkpoint_path, config=None, device=None):
        """
        Initialize the predictor with a trained model checkpoint
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            config: Optional config dict with model parameters
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Default config for VGG_11 (can be overridden)
        if config is None:
            config = {
                'num_classes': 2,
                'in_channel': 3,  # mel + delta + delta2
                'duration': 5,     # seconds
                'delta': True,
                'norm': True,
                'mel_bins': 128
            }
        
        self.config = config
        
        # Initialize model
        self.model = VGG_11(
            num_classes=config['num_classes'],
            in_channel=config['in_channel']
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Using device: {self.device}")
        
        # Class labels
        self.class_labels = ['Normal', 'Abnormal']
    
    def extract_features(self, audio_path):
        """
        Extract mel-spectrogram features from an audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed feature array
        """
        # Read audio with filtering
        audio, sr = read_audio(audio_path, filter=True)
        
        # Extract mel-spectrogram
        feature = LogMelExtractor(audio, sr, mel_bins=self.config['mel_bins'], log=True, snv=False)
        
        # Apply normalization if required
        if self.config['norm']:
            feature = standard_normal_variate(feature)
        
        # Add deltas if required
        if self.config['delta']:
            delta = librosa.feature.delta(feature)
            delta_2 = librosa.feature.delta(delta)
            feature = np.concatenate((feature, delta, delta_2), axis=0)
        
        # Pad or crop to fixed duration
        cycle_len = int(self.config['duration'] * 1000 / 15)  # hop_length = 15ms
        mel_bins, num_frames = feature.shape
        
        if num_frames >= cycle_len:
            # Use center crop for inference
            start_ind = int((num_frames - cycle_len) / 2)
            feature_pad = feature[:, start_ind:start_ind + cycle_len]
        elif num_frames < cycle_len:
            # Pad with wrapping
            feature_pad = np.pad(feature, ((0, 0), (0, cycle_len - num_frames)), mode='wrap')
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
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(feature).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(feature_tensor)
            probabilities = torch.exp(output)  # Convert log probabilities to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'file': Path(audio_path).name,
            'predicted_class': predicted_class,
            'predicted_label': self.class_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                'Normal': probabilities[0][0].item(),
                'Abnormal': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, audio_paths):
        """
        Make predictions on multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict_file(audio_path)
                results.append(result)
                print(f"✓ {result['file']}: {result['predicted_label']} (confidence: {result['confidence']:.2%})")
            except Exception as e:
                print(f"✗ Error processing {audio_path}: {str(e)}")
                results.append({
                    'file': Path(audio_path).name,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_features(self, h5_path, filenames):
        """
        Make predictions from pre-extracted features in HDF5 file
        
        Args:
            h5_path: Path to HDF5 file with features
            filenames: List of filenames to predict
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        with h5py.File(h5_path, 'r') as h5_file:
            for filename in filenames:
                try:
                    # Load feature
                    feature = h5_file[filename][()]
                    
                    if self.config['norm']:
                        feature = standard_normal_variate(feature)
                    
                    if self.config['delta']:
                        delta = librosa.feature.delta(feature)
                        delta_2 = librosa.feature.delta(delta)
                        feature = np.concatenate((feature, delta, delta_2), axis=0)
                    
                    # Pad or crop
                    cycle_len = int(self.config['duration'] * 1000 / 15)
                    mel_bins, num_frames = feature.shape
                    
                    if num_frames >= cycle_len:
                        start_ind = int((num_frames - cycle_len) / 2)
                        feature_pad = feature[:, start_ind:start_ind + cycle_len]
                    elif num_frames < cycle_len:
                        feature_pad = np.pad(feature, ((0, 0), (0, cycle_len - num_frames)), mode='wrap')
                    else:
                        feature_pad = feature
                    
                    # Convert to tensor and predict
                    feature_tensor = torch.FloatTensor(feature_pad).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        output = self.model(feature_tensor)
                        probabilities = torch.exp(output)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    results.append({
                        'file': filename,
                        'predicted_class': predicted_class,
                        'predicted_label': self.class_labels[predicted_class],
                        'confidence': confidence,
                        'probabilities': {
                            'Normal': probabilities[0][0].item(),
                            'Abnormal': probabilities[0][1].item()
                        }
                    })
                    
                    print(f"✓ {filename}: {self.class_labels[predicted_class]} (confidence: {confidence:.2%})")
                    
                except Exception as e:
                    print(f"✗ Error processing {filename}: {str(e)}")
                    results.append({
                        'file': filename,
                        'error': str(e)
                    })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Classification - Prediction')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to audio file, directory of audio files, or HDF5 feature file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save prediction results (CSV)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--in_channel', type=int, default=3,
                        help='Number of input channels (1: mel only, 3: mel+delta+delta2)')
    parser.add_argument('--duration', type=int, default=5,
                        help='Audio duration in seconds')
    parser.add_argument('--no-delta', action='store_true',
                        help='Disable delta features')
    parser.add_argument('--no-norm', action='store_true',
                        help='Disable normalization')
    
    args = parser.parse_args()
    
    # Prepare config
    config = {
        'num_classes': 2,
        'in_channel': args.in_channel,
        'duration': args.duration,
        'delta': not args.no_delta,
        'norm': not args.no_norm,
        'mel_bins': 128
    }
    
    # Initialize predictor
    predictor = HeartSoundPredictor(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device
    )
    
    # Determine input type and make predictions
    input_path = Path(args.input)
    
    if input_path.suffix == '.h5':
        # HDF5 feature file
        print(f"\nPredicting from HDF5 features: {input_path}")
        with h5py.File(input_path, 'r') as h5_file:
            filenames = list(h5_file.keys())
        print(f"Found {len(filenames)} files in HDF5")
        results = predictor.predict_from_features(str(input_path), filenames)
        
    elif input_path.is_dir():
        # Directory of audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"\nFound {len(audio_files)} audio files in {input_path}")
        results = predictor.predict_batch([str(f) for f in audio_files])
        
    elif input_path.is_file():
        # Single audio file
        print(f"\nPredicting on single file: {input_path}")
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
    
    # Save results if output path specified
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    successful = [r for r in results if 'error' not in r]
    if successful:
        normal_count = sum(1 for r in successful if r['predicted_class'] == 0)
        abnormal_count = sum(1 for r in successful if r['predicted_class'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in successful])
        
        print(f"Total predictions: {len(successful)}")
        print(f"Normal: {normal_count} ({normal_count/len(successful):.1%})")
        print(f"Abnormal: {abnormal_count} ({abnormal_count/len(successful):.1%})")
        print(f"Average confidence: {avg_confidence:.2%}")
    
    failed = [r for r in results if 'error' in r]
    if failed:
        print(f"\nFailed predictions: {len(failed)}")


if __name__ == '__main__':
    main()
