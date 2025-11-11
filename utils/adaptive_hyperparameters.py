"""
Adaptive Hyperparameter Scheduler for YOLOv5 Training
Automatically adjusts hyperparameters based on training progress and validation metrics.
"""

import json
import logging
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class AdaptiveHyperparameterScheduler:
    """
    Adaptive hyperparameter scheduler that adjusts hyperparameters based on training metrics.
    
    Rules:
    - If recall < 0.30 for N consecutive epochs → increase uncertainty_weight
    - If mAP@0.5:0.95 improvement < threshold over N epochs → increase box weight
    - If object loss > 2× box loss for N epochs → reduce anchor_t
    - If pixel loss < threshold → increase uncertainty_weight
    - If validation loss stagnates → trigger learning rate adjustments
    """
    
    def __init__(
        self,
        hyp: Dict,
        save_dir: Path,
        enabled: bool = True,
        check_interval: int = 1,  # Check every N epochs
        min_epochs_before_adapt: int = 10,  # Don't adapt before this epoch
    ):
        """
        Initialize adaptive hyperparameter scheduler.
        
        Args:
            hyp: Hyperparameter dictionary (will be modified in-place)
            save_dir: Directory to save adaptive hyperparameter logs
            enabled: Whether adaptive tuning is enabled
            check_interval: Check conditions every N epochs
            min_epochs_before_adapt: Minimum epochs before starting adaptation
        """
        self.hyp = hyp
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        self.check_interval = check_interval
        self.min_epochs_before_adapt = min_epochs_before_adapt
        
        # History tracking
        self.history: List[Dict] = []
        self.metric_history: deque = deque(maxlen=20)  # Keep last 20 epochs
        
        # Adaptive parameter bounds
        self.bounds = {
            'uncertainty_weight': (0.1, 0.3),
            'box': (0.05, 0.1),
            'obj': (0.8, 1.2),
            'cls': (0.3, 0.7),
            'anchor_t': (2.0, 5.0),
            'lrf': (0.01, 0.2),  # Learning rate final
        }
        
        # Step sizes for adjustments
        self.step_sizes = {
            'uncertainty_weight': 0.05,
            'box': 0.025,
            'obj': 0.05,
            'cls': 0.05,
            'anchor_t': 0.5,
            'lrf': 0.02,
        }
        
        # Rule configurations
        self.rules = {
            'recall_threshold': 0.30,
            'recall_check_epochs': 5,
            'map_improvement_threshold': 0.01,
            'map_check_epochs': 10,
            'obj_box_ratio_threshold': 2.0,
            'obj_box_check_epochs': 10,
            'pixel_loss_threshold': 0.008,
            'val_loss_stagnation_threshold': 0.001,
            'val_loss_check_epochs': 10,
        }
        
        # Initialize log file
        self.log_file = self.save_dir / 'adaptive_hyp_log.yaml'
        self.json_log_file = self.save_dir / 'adaptive_hyp_log.json'
        
        # Track initial values
        self.initial_hyp = hyp.copy()
        
        if self.enabled:
            logger.info("Adaptive hyperparameter scheduler initialized")
            logger.info(f"Initial hyperparameters: {self._get_tracked_params()}")
    
    def _get_tracked_params(self) -> Dict:
        """Get current values of tracked hyperparameters."""
        return {
            'uncertainty_weight': self.hyp.get('uncertainty_weight', 0.0),
            'box': self.hyp.get('box', 0.0),
            'obj': self.hyp.get('obj', 0.0),
            'cls': self.hyp.get('cls', 0.0),
            'anchor_t': self.hyp.get('anchor_t', 0.0),
            'lrf': self.hyp.get('lrf', 0.0),
        }
    
    def _clip_value(self, param_name: str, value: float) -> float:
        """Clip value to safe bounds."""
        if param_name in self.bounds:
            min_val, max_val = self.bounds[param_name]
            return np.clip(value, min_val, max_val)
        return value
    
    def _apply_adjustment(self, param_name: str, adjustment: float, reason: str, epoch: int) -> bool:
        """
        Apply hyperparameter adjustment with bounds checking.
        
        Returns:
            True if adjustment was applied, False if clipped
        """
        if param_name not in self.hyp:
            logger.warning(f"Parameter {param_name} not found in hyperparameters")
            return False
        
        old_value = self.hyp[param_name]
        new_value = self._clip_value(param_name, old_value + adjustment)
        
        if abs(new_value - old_value) < 1e-6:
            logger.debug(f"Adjustment for {param_name} clipped (would be {old_value + adjustment:.6f})")
            return False
        
        self.hyp[param_name] = new_value
        
        change_log = {
            'epoch': epoch,
            'parameter': param_name,
            'old_value': float(old_value),
            'new_value': float(new_value),
            'adjustment': float(adjustment),
            'reason': reason,
        }
        
        self.history.append(change_log)
        
        logger.info(
            f"Epoch {epoch}: Adjusted {param_name}: {old_value:.6f} → {new_value:.6f} "
            f"(reason: {reason})"
        )
        
        return True
    
    def _check_recall_rule(self, metrics: Dict, epoch: int) -> Optional[Tuple[str, float, str]]:
        """Check if recall is low for consecutive epochs."""
        if len(self.metric_history) < self.rules['recall_check_epochs']:
            return None
        
        recent_recalls = [m.get('recall', 0) for m in list(self.metric_history)[-self.rules['recall_check_epochs']:]]
        
        if all(r < self.rules['recall_threshold'] for r in recent_recalls):
            current_recall = metrics.get('recall', 0)
            if current_recall < self.rules['recall_threshold']:
                adjustment = self.step_sizes['uncertainty_weight']
                reason = f"Recall < {self.rules['recall_threshold']} for {self.rules['recall_check_epochs']} epochs (current: {current_recall:.4f})"
                return ('uncertainty_weight', adjustment, reason)
        
        return None
    
    def _check_map_improvement_rule(self, metrics: Dict, epoch: int) -> Optional[Tuple[str, float, str]]:
        """Check if mAP@0.5:0.95 improvement is too slow."""
        if len(self.metric_history) < self.rules['map_check_epochs']:
            return None
        
        recent_maps = [m.get('map50_95', 0) for m in list(self.metric_history)[-self.rules['map_check_epochs']:]]
        
        if len(recent_maps) >= 2:
            improvement = recent_maps[-1] - recent_maps[0]
            if improvement < self.rules['map_improvement_threshold']:
                adjustment = self.step_sizes['box']
                reason = f"mAP@0.5:0.95 improvement < {self.rules['map_improvement_threshold']:.4f} over {self.rules['map_check_epochs']} epochs (improvement: {improvement:.4f})"
                return ('box', adjustment, reason)
        
        return None
    
    def _check_obj_box_ratio_rule(self, metrics: Dict, epoch: int) -> Optional[Tuple[str, float, str]]:
        """Check if object loss is too high relative to box loss."""
        if len(self.metric_history) < self.rules['obj_box_check_epochs']:
            return None
        
        recent_ratios = []
        for m in list(self.metric_history)[-self.rules['obj_box_check_epochs']:]:
            obj_loss = m.get('obj_loss', 0)
            box_loss = m.get('box_loss', 0)
            if box_loss > 0:
                recent_ratios.append(obj_loss / box_loss)
        
        if len(recent_ratios) >= self.rules['obj_box_check_epochs']:
            avg_ratio = np.mean(recent_ratios)
            if avg_ratio > self.rules['obj_box_ratio_threshold']:
                adjustment = -self.step_sizes['anchor_t']  # Reduce anchor_t
                reason = f"Object loss > {self.rules['obj_box_ratio_threshold']}× box loss for {self.rules['obj_box_check_epochs']} epochs (avg ratio: {avg_ratio:.2f})"
                return ('anchor_t', adjustment, reason)
        
        return None
    
    def _check_pixel_loss_rule(self, metrics: Dict, epoch: int) -> Optional[Tuple[str, float, str]]:
        """Check if pixel loss is too small."""
        pixel_loss = metrics.get('pixl_loss', float('inf'))
        
        if pixel_loss < self.rules['pixel_loss_threshold']:
            adjustment = self.step_sizes['uncertainty_weight']
            reason = f"Pixel loss < {self.rules['pixel_loss_threshold']:.4f} (current: {pixel_loss:.6f})"
            return ('uncertainty_weight', adjustment, reason)
        
        return None
    
    def _check_val_loss_stagnation_rule(self, metrics: Dict, epoch: int) -> Optional[Tuple[str, float, str]]:
        """Check if validation loss has stagnated."""
        if len(self.metric_history) < self.rules['val_loss_check_epochs']:
            return None
        
        recent_val_losses = [m.get('val_total_loss', float('inf')) for m in list(self.metric_history)[-self.rules['val_loss_check_epochs']:]]
        
        if len(recent_val_losses) >= 2:
            # Check if loss is not improving
            best_loss = min(recent_val_losses)
            current_loss = recent_val_losses[-1]
            
            if abs(current_loss - best_loss) < self.rules['val_loss_stagnation_threshold']:
                # Reduce learning rate final to allow longer training
                adjustment = -self.step_sizes['lrf']  # Reduce lrf (allows higher final LR)
                reason = f"Validation loss stagnated (improvement < {self.rules['val_loss_stagnation_threshold']:.4f} over {self.rules['val_loss_check_epochs']} epochs)"
                return ('lrf', adjustment, reason)
        
        return None
    
    def update(self, epoch: int, metrics: Dict) -> Dict:
        """
        Update hyperparameters based on current metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing:
                - recall: Validation recall
                - map50_95: mAP@0.5:0.95
                - box_loss: Training box loss
                - obj_loss: Training object loss
                - pixl_loss: Training pixel loss
                - val_total_loss: Validation total loss
                - (and other metrics)
        
        Returns:
            Dictionary of applied adjustments
        """
        if not self.enabled:
            return {}
        
        if epoch < self.min_epochs_before_adapt:
            return {}
        
        if epoch % self.check_interval != 0:
            return {}
        
        # Add current metrics to history
        self.metric_history.append(metrics.copy())
        
        applied_adjustments = {}
        
        # Apply rules in priority order
        rules_to_check = [
            self._check_pixel_loss_rule,  # Check first (most specific)
            self._check_recall_rule,
            self._check_map_improvement_rule,
            self._check_obj_box_ratio_rule,
            self._check_val_loss_stagnation_rule,
        ]
        
        for rule_func in rules_to_check:
            result = rule_func(metrics, epoch)
            if result is not None:
                param_name, adjustment, reason = result
                
                # Only apply one adjustment per epoch to avoid conflicts
                if param_name not in applied_adjustments:
                    if self._apply_adjustment(param_name, adjustment, reason, epoch):
                        applied_adjustments[param_name] = {
                            'adjustment': adjustment,
                            'reason': reason,
                            'new_value': self.hyp[param_name],
                        }
                        break  # Apply only one rule per epoch
        
        # Save log after each update
        if applied_adjustments:
            self._save_log(epoch)
        
        return applied_adjustments
    
    def _save_log(self, epoch: int):
        """Save adaptive hyperparameter log to file."""
        log_data = {
            'epoch': epoch,
            'current_hyperparameters': self._get_tracked_params(),
            'initial_hyperparameters': {k: float(v) for k, v in self.initial_hyp.items() if k in self._get_tracked_params()},
            'adjustment_history': self.history[-10:],  # Last 10 adjustments
        }
        
        # Save as YAML
        try:
            with open(self.log_file, 'w') as f:
                yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to save YAML log: {e}")
        
        # Save as JSON (for easier parsing)
        try:
            with open(self.json_log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save JSON log: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of adaptive hyperparameter changes."""
        if not self.history:
            return {
                'enabled': self.enabled,
                'total_adjustments': 0,
                'current_params': self._get_tracked_params(),
            }
        
        return {
            'enabled': self.enabled,
            'total_adjustments': len(self.history),
            'current_params': self._get_tracked_params(),
            'initial_params': {k: float(v) for k, v in self.initial_hyp.items() if k in self._get_tracked_params()},
            'recent_adjustments': self.history[-5:],  # Last 5 adjustments
        }
    
    def plot_evolution(self, save_path: Optional[Path] = None):
        """Plot hyperparameter evolution over epochs."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        if not self.history:
            logger.info("No hyperparameter adjustments to plot")
            return
        
        # Extract data
        epochs = [h['epoch'] for h in self.history]
        params = {}
        for h in self.history:
            param = h['parameter']
            if param not in params:
                params[param] = {'epochs': [], 'values': []}
            params[param]['epochs'].append(h['epoch'])
            params[param]['values'].append(h['new_value'])
        
        # Create plot
        fig, axes = plt.subplots(len(params), 1, figsize=(10, 3 * len(params)))
        if len(params) == 1:
            axes = [axes]
        
        for idx, (param_name, data) in enumerate(params.items()):
            ax = axes[idx]
            ax.plot(data['epochs'], data['values'], 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Evolution')
            ax.grid(True, alpha=0.3)
            
            # Add initial value line
            if param_name in self.initial_hyp:
                ax.axhline(y=self.initial_hyp[param_name], color='r', linestyle='--', 
                          alpha=0.5, label=f'Initial: {self.initial_hyp[param_name]:.4f}')
                ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Hyperparameter evolution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

