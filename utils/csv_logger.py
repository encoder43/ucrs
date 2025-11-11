"""
CSV Logger for training metrics
Saves all training and validation metrics to CSV file for easy analysis
"""

import csv
from pathlib import Path
from typing import List, Dict, Any


class CSVLogger:
    """Logger that saves metrics to CSV file"""
    
    def __init__(self, save_dir: Path, filename: str = 'metrics.csv'):
        """
        Initialize CSV logger
        
        Args:
            save_dir: Directory to save CSV file
            filename: Name of CSV file
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.save_dir / filename
        
        # Define column headers
        self.headers = [
            'epoch',
            # Training losses
            'train_box_loss', 'train_obj_loss', 'train_cls_loss',
            'train_pixl_loss', 'train_area_loss', 'train_dist_loss', 'train_total_loss',
            # Validation metrics
            'val_precision', 'val_recall', 'val_map50', 'val_map50_95',
            'val_bpr', 'val_occupy',
            # Validation losses
            'val_box_loss', 'val_obj_loss', 'val_cls_loss',
            'val_pixl_loss', 'val_area_loss', 'val_dist_loss',
            # Learning rates
            'lr0', 'lr1', 'lr2',
            # Fitness (combined metric)
            'fitness',
            # Performance metrics
            'gflops', 'fps'
        ]
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def log(self, epoch: int, train_losses: List[float], val_metrics: List[float], 
            val_losses: List[float], learning_rates: List[float], fitness: float = 0.0,
            gflops: float = 0.0, fps: float = 0.0):
        """
        Log metrics for one epoch
        
        Args:
            epoch: Current epoch number (0-indexed)
            train_losses: [box, obj, cls, pixl, area, dist, total] training losses
            val_metrics: [precision, recall, map50, map50_95, bpr, occupy] validation metrics
            val_losses: [box, obj, cls, pixl, area, dist] validation losses
            learning_rates: [lr0, lr1, lr2] learning rates for different parameter groups
            fitness: Combined fitness score
            gflops: GFLOPs (Giga Floating Point Operations per Second)
            fps: FPS (Frames Per Second) - inference speed
        """
        row = [
            epoch + 1,  # 1-indexed epoch for readability
            *train_losses[:7],  # All 7 training losses (including total)
            *val_metrics[:6],  # Validation metrics
            *val_losses[:6],  # Validation losses
            *learning_rates[:3],  # Learning rates
            fitness,
            gflops,  # GFLOPs
            fps  # FPS
        ]
        
        # Ensure row has correct length
        if len(row) < len(self.headers):
            row.extend([0.0] * (len(self.headers) - len(row)))
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def __repr__(self):
        return f"CSVLogger(save_dir={self.save_dir}, csv_path={self.csv_path})"

