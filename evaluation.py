# evaluation.py - Evaluation metrics and reporting
import csv
import json
from typing import List, Dict, Tuple
from utils import calculate_iou
from config import IOU_THRESHOLD, TARGET_CLASSES

class EvaluationMetrics:
    def __init__(self, iou_threshold: float = IOU_THRESHOLD):
        self.iou_threshold = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
    
    def match_boxes(self, ground_truth: List[Dict], predictions: List[Dict]) -> Tuple[int, int, int]:
        """
        Match predicted boxes to ground truth using IoU.
        Returns: (TP, FP, FN)
        """
        tp, fp, fn = 0, 0, 0
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find best matching GT
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            pred_box = (pred['x1'], pred['y1'], pred['x2'], pred['y2'])
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt or gt['class_id'] != pred['class_id']:
                    continue
                gt_box = (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                fp += 1
        
        fn = len(ground_truth) - len(matched_gt)
        
        return tp, fp, fn
    
    def calculate_metrics(self, tp: int, fp: int, fn: int) -> Dict:
        """Calculate precision, recall, F1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }

class EvaluationReporter:
    def __init__(self, output_csv: str = "evaluation/results.csv"):
        self.output_csv = output_csv
        self.results = []
    
    def add_result(self, 
                   image_id: str,
                   dataset_name: str,
                   class_name: str,
                   tp: int, fp: int, fn: int,
                   precision: float, recall: float, f1: float,
                   conf_threshold: float,
                   iou_threshold: float):
        """Add evaluation result."""
        self.results.append({
            'image_id': image_id,
            'dataset': dataset_name,
            'class': class_name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        })
    
    def save_csv(self):
        """Save results to CSV."""
        if not self.results:
            return
        import os
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
        keys = self.results[0].keys()
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"✅ Evaluation saved to {self.output_csv}")
    
    def summary_stats(self) -> Dict:
        """Calculate aggregate stats across all results."""
        if not self.results:
            return {}
        
        total_tp = sum(r['TP'] for r in self.results)
        total_fp = sum(r['FP'] for r in self.results)
        total_fn = sum(r['FN'] for r in self.results)
        
        avg_precision = sum(r['precision'] for r in self.results) / len(self.results)
        avg_recall = sum(r['recall'] for r in self.results) / len(self.results)
        avg_f1 = sum(r['f1'] for r in self.results) / len(self.results)
        
        return {
            'total_images': len(self.results),
            'total_TP': total_tp,
            'total_FP': total_fp,
            'total_FN': total_fn,
            'avg_precision': round(avg_precision, 4),
            'avg_recall': round(avg_recall, 4),
            'avg_f1': round(avg_f1, 4)
        }
