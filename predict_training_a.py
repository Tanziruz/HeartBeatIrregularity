"""
Predict on all audio files in the training-a dataset
"""
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

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



def load_ground_truth(label_csv_path, dataset_folder):
    """
    Load ground truth labels for the dataset
    
    Args:
        label_csv_path: Path to label.csv file
        dataset_folder: Folder name (e.g., 'training-a')
        
    Returns:
        Dictionary mapping filename to label
    """
    df = pd.read_csv(label_csv_path)
    
    # Filter for the specific dataset folder
    df_filtered = df[df['filename'].str.contains(dataset_folder)]
    
    # Create mapping from basename to label
    ground_truth = {}
    for _, row in df_filtered.iterrows():
        filename = Path(row['filename']).name
        ground_truth[filename] = int(row['label'])
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description='Predict on entire training-a dataset')
    parser.add_argument('--checkpoint', type=str, default='model_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, 
                        default='data/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/training-a',
                        help='Path to training-a folder')
    parser.add_argument('--label_csv', type=str, default='data/label.csv',
                        help='Path to label CSV file for ground truth')
    parser.add_argument('--output', type=str, default='predictions_training_a.csv',
                        help='Output CSV file for predictions')
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for prediction (currently only supports 1)')
    
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
    print("Loading model...")
    predictor = HeartSoundPredictor(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device
    )
    
    # Find all audio files in training-a
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    audio_files = sorted(list(dataset_path.glob('*.wav')))
    print(f"\nFound {len(audio_files)} audio files in {dataset_path}")
    
    if len(audio_files) == 0:
        print("No .wav files found!")
        return
    
    # Load ground truth if available
    ground_truth = {}
    if Path(args.label_csv).exists():
        ground_truth = load_ground_truth(args.label_csv, dataset_path.name)
        print(f"Loaded ground truth for {len(ground_truth)} files")
    
    # Make predictions
    print("\nMaking predictions...")
    results = []
    y_true = []
    y_pred = []
    
    for audio_file in tqdm(audio_files, desc="Predicting"):
        try:
            result = predictor.predict_file(str(audio_file))
            
            # Add ground truth if available
            filename = audio_file.name
            if filename in ground_truth:
                result['true_label'] = ground_truth[filename]
                result['true_label_name'] = predictor.class_labels[ground_truth[filename]]
                result['correct'] = result['predicted_class'] == ground_truth[filename]
                y_true.append(ground_truth[filename])
                y_pred.append(result['predicted_class'])
            
            results.append(result)
            
        except Exception as e:
            print(f"\nError processing {audio_file.name}: {str(e)}")
            results.append({
                'file': audio_file.name,
                'error': str(e)
            })
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\n✓ Predictions saved to: {args.output}")
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"\nTotal files: {len(audio_files)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed predictions: {len(failed)}")
    
    if successful:
        normal_count = sum(1 for r in successful if r['predicted_class'] == 0)
        abnormal_count = sum(1 for r in successful if r['predicted_class'] == 1)
        avg_confidence = np.mean([r['confidence'] for r in successful])
        
        print(f"\nPredicted Distribution:")
        print(f"  Normal: {normal_count} ({normal_count/len(successful)*100:.1f}%)")
        print(f"  Abnormal: {abnormal_count} ({abnormal_count/len(successful)*100:.1f}%)")
        print(f"\nAverage confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    # If ground truth is available, calculate metrics
    if len(y_true) > 0 and len(y_pred) > 0:
        print(f"\n{'='*70}")
        print(f"EVALUATION METRICS (vs Ground Truth)")
        print(f"{'='*70}")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Normal  Abnormal")
        print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:8d}")
        print(f"       Abnormal {cm[1][0]:6d}  {cm[1][1]:8d}")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for i, label in enumerate(predictor.class_labels):
            print(f"\n{label}:")
            print(f"  Precision: {precision_pc[i]:.4f}")
            print(f"  Recall: {recall_pc[i]:.4f}")
            print(f"  F1 Score: {f1_pc[i]:.4f}")
            print(f"  Support: {support_pc[i]}")
        
        # Classification report
        print(f"\n{'='*70}")
        print("Detailed Classification Report:")
        print(f"{'='*70}")
        print(classification_report(y_true, y_pred, target_names=predictor.class_labels))
        
        # Correctly/incorrectly classified
        correct = sum(1 for r in successful if r.get('correct', False))
        incorrect = len(y_true) - correct
        print(f"\nCorrectly classified: {correct}/{len(y_true)} ({correct/len(y_true)*100:.2f}%)")
        print(f"Incorrectly classified: {incorrect}/{len(y_true)} ({incorrect/len(y_true)*100:.2f}%)")
        
        # Show some misclassified examples
        misclassified = [r for r in successful if r.get('correct', True) == False]
        if misclassified:
            print(f"\n{'='*70}")
            print(f"Sample Misclassified Examples (first 10):")
            print(f"{'='*70}")
            for r in misclassified[:10]:
                print(f"\n{r['file']}:")
                print(f"  True: {r['true_label_name']}, Predicted: {r['predicted_label']}")
                print(f"  Confidence: {r['confidence']:.4f}")
                print(f"  Probabilities: Normal={r['probabilities']['Normal']:.4f}, "
                      f"Abnormal={r['probabilities']['Abnormal']:.4f}")
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
