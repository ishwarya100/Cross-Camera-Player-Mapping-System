import cv2
import numpy as np
from ultralytics import YOLO
import json
import logging
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define BASE_DIR at module level
BASE_DIR = r"C:\Users\user\OneDrive\Desktop\Stealthmode_Assignment"

@dataclass
class PlayerDetection:
    """Enhanced detection data structure"""
    frame_idx: int
    bbox: np.ndarray
    confidence: float
    center: np.ndarray
    features: np.ndarray
    class_id: int
    timestamp: float

@dataclass
class PlayerTrack:
    """Enhanced track data structure"""
    id: int
    detections: List[PlayerDetection]
    feature_history: List[np.ndarray]
    center_history: List[np.ndarray]
    avg_features: np.ndarray
    last_seen: int
    confidence_score: float

class EnhancedPlayerMapper:
    def __init__(self, model_path: str):
        logger.info(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.broadcast_detections = []
        self.tacticam_detections = []
        self.broadcast_tracks = []
        self.tacticam_tracks = []
        self.player_mappings = {}
        self.video_metadata = {}
        self.broadcast_frames = []
        self.tacticam_frames = []
        logger.info(f"Model loaded! Classes: {self.model.names}")
        
    def extract_enhanced_features(self, frame: np.ndarray, bbox: np.ndarray, bins: int = 32) -> np.ndarray:
        """Extract comprehensive visual features for player identification"""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bounding box: x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2)
            return np.zeros(bins * 6 + 8)
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            logger.warning("Empty ROI for bbox: x1=%d, y1=%d, x2=%d, y2=%d", x1, y1, x2, y2)
            return np.zeros(bins * 6 + 8)
        
        features = []
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        features.extend([
            cv2.normalize(hist_h, hist_h).flatten(),
            cv2.normalize(hist_s, hist_s).flatten(),
            cv2.normalize(hist_v, hist_v).flatten()
        ])
        
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [bins], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [bins], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [bins], [0, 256])
        
        features.extend([
            cv2.normalize(hist_l, hist_l).flatten(),
            cv2.normalize(hist_a, hist_a).flatten(),
            cv2.normalize(hist_b, hist_b).flatten()
        ])
        
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        frame_h, frame_w = frame.shape[:2]
        norm_width = width / frame_w
        norm_height = height / frame_h
        norm_area = area / (frame_w * frame_h)
        norm_center_x = bbox_center_x / frame_w
        norm_center_y = bbox_center_y / frame_h
        
        shape_features = np.array([
            norm_width, norm_height, aspect_ratio, norm_area,
            norm_center_x, norm_center_y, width, height
        ])
        
        color_features = np.concatenate(features)
        all_features = np.concatenate([color_features, shape_features])
        
        return all_features
    
    def process_video_enhanced(self, video_path: str, camera_type: str, max_frames: int = 1000) -> List[List[PlayerDetection]]:
        """Enhanced video processing with frame storage for visualization"""
        if not os.path.exists(video_path):
            logger.error("Video not found: %s", video_path)
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.video_metadata[camera_type] = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0
        }
        
        logger.info("%s: %d frames, %.2f FPS, %.2fs duration", 
                    camera_type.upper(), total_frames, fps, total_frames/fps)
        
        detections = []
        frames = []
        frame_count = 0
        total_detections = 0
        frame_step = max(1, total_frames // max_frames)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                try:
                    # Store frame for visualization
                    frames.append(frame.copy())
                    results = self.model(frame, verbose=False)
                    frame_detections = []
                    timestamp = frame_count / fps if fps > 0 else frame_count
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls)
                                if cls_id in [1, 2]:  # players and goalkeepers
                                    bbox = box.xyxy[0].cpu().numpy()
                                    confidence = float(box.conf.cpu().numpy()[0])
                                    
                                    if confidence < 0.3:
                                        continue
                                    
                                    features = self.extract_enhanced_features(frame, bbox)
                                    center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                                    
                                    detection = PlayerDetection(
                                        frame_idx=frame_count,
                                        bbox=bbox,
                                        confidence=confidence,
                                        center=center,
                                        features=features,
                                        class_id=cls_id,
                                        timestamp=timestamp
                                    )
                                    
                                    frame_detections.append(detection)
                                    total_detections += 1
                    
                    detections.append(frame_detections)
                    
                    if len(detections) % 100 == 0:
                        logger.debug("Processed %d frames, %d detections", len(detections), total_detections)
                        
                except Exception as e:
                    logger.error("Error processing frame %d: %s", frame_count, e)
                    detections.append([])
            
            frame_count += 1
        
        cap.release()
        
        # Store frames
        if camera_type == 'broadcast':
            self.broadcast_frames = frames
        else:
            self.tacticam_frames = frames
            
        logger.info("Final: %d sampled frames, %d total detections", len(detections), total_detections)
        return detections
    
    def track_players_enhanced(self, detections: List[List[PlayerDetection]], 
                             max_distance: float = 100, 
                             max_frames_gap: int = 10) -> List[PlayerTrack]:
        """Enhanced tracking with better association and track management"""
        if not detections:
            logger.warning("No detections provided for tracking")
            return []
            
        tracks = []
        next_id = 0
        
        for frame_idx, frame_detections in enumerate(detections):
            if frame_idx == 0:
                for det in frame_detections:
                    track = PlayerTrack(
                        id=next_id,
                        detections=[det],
                        feature_history=[det.features],
                        center_history=[det.center],
                        avg_features=det.features.copy(),
                        last_seen=frame_idx,
                        confidence_score=det.confidence
                    )
                    tracks.append(track)
                    next_id += 1
            else:
                active_tracks = [t for t in tracks if (frame_idx - t.last_seen) <= max_frames_gap]
                
                if not active_tracks or not frame_detections:
                    for det in frame_detections:
                        track = PlayerTrack(
                            id=next_id,
                            detections=[det],
                            feature_history=[det.features],
                            center_history=[det.center],
                            avg_features=det.features.copy(),
                            last_seen=frame_idx,
                            confidence_score=det.confidence
                        )
                        tracks.append(track)
                        next_id += 1
                    continue
                
                cost_matrix = np.zeros((len(active_tracks), len(frame_detections)))
                
                for i, track in enumerate(active_tracks):
                    for j, det in enumerate(frame_detections):
                        spatial_dist = np.linalg.norm(track.center_history[-1] - det.center)
                        feature_dist = 1 - self.calculate_feature_similarity(track.avg_features, det.features)
                        temporal_penalty = (frame_idx - track.last_seen) * 0.1
                        cost = spatial_dist + feature_dist * 50 + temporal_penalty
                        cost_matrix[i, j] = cost
                
                if cost_matrix.size > 0:
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    matched_tracks = set()
                    matched_detections = set()
                    
                    for i, j in zip(row_indices, col_indices):
                        if cost_matrix[i, j] < max_distance:
                            track = active_tracks[i]
                            det = frame_detections[j]
                            track.detections.append(det)
                            track.feature_history.append(det.features)
                            track.center_history.append(det.center)
                            track.last_seen = frame_idx
                            alpha = 0.1 if len(track.feature_history) > 10 else 0.3
                            track.avg_features = (1-alpha) * track.avg_features + alpha * det.features
                            track.confidence_score = (track.confidence_score + det.confidence) / 2
                            matched_tracks.add(i)
                            matched_detections.add(j)
                    
                    for j, det in enumerate(frame_detections):
                        if j not in matched_detections:
                            track = PlayerTrack(
                                id=next_id,
                                detections=[det],
                                feature_history=[det.features],
                                center_history=[det.center],
                                avg_features=det.features.copy(),
                                last_seen=frame_idx,
                                confidence_score=det.confidence
                            )
                            tracks.append(track)
                            next_id += 1
        
        good_tracks = []
        for track in tracks:
            min_detections = 5
            min_confidence = 0.4
            if (len(track.detections) >= min_detections and 
                track.confidence_score >= min_confidence):
                good_tracks.append(track)
        
        logger.info("Found %d quality tracks from %d total tracks", len(good_tracks), len(tracks))
        return good_tracks
    
    def calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate sophisticated feature similarity"""
        if len(features1) != len(features2):
            logger.warning("Feature length mismatch: %d vs %d", len(features1), len(features2))
            return 0.0
        
        color_size = len(features1) - 8
        color1, shape1 = features1[:color_size], features1[color_size:]
        color2, shape2 = features2[:color_size], features2[color_size:]
        
        bins = color_size // 6
        hsv1, lab1 = color1[:bins*3], color1[bins*3:]
        hsv2, lab2 = color2[:bins*3], color2[bins*3:]
        
        hsv_corr = np.corrcoef(hsv1, hsv2)[0, 1] if len(hsv1) > 1 else 0
        if np.isnan(hsv_corr):
            hsv_corr = 0
        
        lab_corr = np.corrcoef(lab1, lab2)[0, 1] if len(lab1) > 1 else 0
        if np.isnan(lab_corr):
            lab_corr = 0
        
        def chi_square_similarity(h1, h2):
            h1_norm = h1 / (np.sum(h1) + 1e-10)
            h2_norm = h2 / (np.sum(h2) + 1e-10)
            chi2 = 0.5 * np.sum((h1_norm - h2_norm)**2 / (h1_norm + h2_norm + 1e-10))
            return 1 / (1 + chi2)
        
        hsv_chi2 = chi_square_similarity(hsv1, hsv2)
        lab_chi2 = chi_square_similarity(lab1, lab2)
        
        color_sim = (abs(hsv_corr) * 0.3 + abs(lab_corr) * 0.3 + 
                    hsv_chi2 * 0.2 + lab_chi2 * 0.2)
        
        shape_diff = np.abs(shape1 - shape2)
        shape_weights = np.array([0.2, 0.2, 0.3, 0.1, 0.1, 0.05, 0.025, 0.025])
        weighted_shape_diff = shape_diff * shape_weights
        shape_sim = 1 / (1 + np.sum(weighted_shape_diff))
        
        total_similarity = 0.7 * color_sim + 0.3 * shape_sim
        return max(0, min(1, total_similarity))
    
    def generate_animated_overlays(self, output_path: str = os.path.join(BASE_DIR, 'matching_animation.gif'), duration: int = 10):
        """Generate animated GIF showing synchronized camera views with player tracking lines"""
        if not self.broadcast_frames or not self.tacticam_frames or not self.player_mappings:
            logger.warning("Cannot generate animation: missing frames or mappings")
            return
        
        logger.info("Generating animated matching overlays...")
        
        # Ensure frames are synchronized
        min_frames = min(len(self.broadcast_frames), len(self.tacticam_frames))
        frames = []
        
        # Colors for different players
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        for frame_idx in range(min_frames):
            bc_frame = self.broadcast_frames[frame_idx].copy()
            tac_frame = self.tacticam_frames[frame_idx].copy()
            
            # Draw tracks on both frames
            for tac_id, bc_id in self.player_mappings.items():
                tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
                bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
                
                if not tac_track or not bc_track:
                    continue
                
                # Find detections for this frame
                tac_det = next((d for d in tac_track.detections if d.frame_idx == frame_idx), None)
                bc_det = next((d for d in bc_track.detections if d.frame_idx == frame_idx), None)
                
                color = colors[tac_id % len(colors)]
                
                if tac_det:
                    x1, y1, x2, y2 = map(int, tac_det.bbox)
                    cv2.rectangle(tac_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(tac_frame, f"T{tac_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if bc_det:
                    x1, y1, x2, y2 = map(int, bc_det.bbox)
                    cv2.rectangle(bc_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(bc_frame, f"B{bc_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Combine frames side by side
            tac_h, tac_w = tac_frame.shape[:2]
            bc_h, bc_w = bc_frame.shape[:2]
            max_h = max(tac_h, bc_h)
            combined = np.zeros((max_h, tac_w + bc_w, 3), dtype=np.uint8)
            
            # Resize frames to match height
            tac_frame = cv2.resize(tac_frame, (tac_w, max_h))
            bc_frame = cv2.resize(bc_frame, (bc_w, max_h))
            
            combined[:, :tac_w] = tac_frame
            combined[:, tac_w:] = bc_frame
            
            # Add labels
            cv2.putText(combined, "Tacticam", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Broadcast", (tac_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Convert to RGB for imageio
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            frames.append(combined_rgb)
        
        # Save as GIF
        try:
            imageio.mimsave(output_path, frames, duration=duration/len(frames))
            logger.info("Animated overlay saved to %s", output_path)
        except Exception as e:
            logger.error("Error saving GIF: %s", e)
    
    def generate_before_after_comparison(self, output_path: str = os.path.join(BASE_DIR, 'before_after_mapping.mp4')):
        """Generate video showing before and after mapping with error corrections"""
        if not self.broadcast_frames or not self.tacticam_frames or not self.player_mappings:
            logger.warning("Cannot generate comparison: missing frames or mappings")
            return
        
        logger.info("Generating before/after mapping comparison...")
        
        # Initial matching based on naive feature similarity
        initial_mappings = {}
        similarities = np.zeros((len(self.tacticam_tracks), len(self.broadcast_tracks)))
        for i, tac_track in enumerate(self.tacticam_tracks):
            for j, bc_track in enumerate(self.broadcast_tracks):
                similarities[i, j] = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
        
        initial_cost = 1 - similarities
        row_indices, col_indices = linear_sum_assignment(initial_cost)
        for i, j in zip(row_indices, col_indices):
            if similarities[i, j] > 0.2:  # Loose threshold for initial matching
                initial_mappings[self.tacticam_tracks[i].id] = self.broadcast_tracks[j].id
        
        # Create video writer
        min_frames = min(len(self.broadcast_frames), len(self.tacticam_frames))
        tac_h, tac_w = self.tacticam_frames[0].shape[:2]
        bc_h, bc_w = self.broadcast_frames[0].shape[:2]
        max_h = max(tac_h, bc_h)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (tac_w + bc_w, max_h))
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        for frame_idx in range(min_frames):
            bc_frame = self.broadcast_frames[frame_idx].copy()
            tac_frame = self.tacticam_frames[frame_idx].copy()
            
            # Draw initial (before) mappings
            for tac_id, bc_id in initial_mappings.items():
                tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
                bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
                
                if not tac_track or not bc_track:
                    continue
                
                tac_det = next((d for d in tac_track.detections if d.frame_idx == frame_idx), None)
                bc_det = next((d for d in bc_track.detections if d.frame_idx == frame_idx), None)
                
                color = colors[tac_id % len(colors)]
                
                if tac_det:
                    x1, y1, x2, y2 = map(int, tac_det.bbox)
                    cv2.rectangle(tac_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(tac_frame, f"T{tac_id}(Init)", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if bc_det:
                    x1, y1, x2, y2 = map(int, bc_det.bbox)
                    cv2.rectangle(bc_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(bc_frame, f"B{bc_id}(Init)", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw final (after) mappings
            for tac_id, bc_id in self.player_mappings.items():
                tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
                bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
                
                if not tac_track or not bc_track:
                    continue
                
                tac_det = next((d for d in tac_track.detections if d.frame_idx == frame_idx), None)
                bc_det = next((d for d in bc_track.detections if d.frame_idx == frame_idx), None)
                
                color = colors[tac_id % len(colors)]
                
                if tac_det:
                    x1, y1, x2, y2 = map(int, tac_det.bbox)
                    cv2.rectangle(tac_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(tac_frame, f"T{tac_id}(Final)", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if bc_det:
                    x1, y1, x2, y2 = map(int, bc_det.bbox)
                    cv2.rectangle(bc_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(bc_frame, f"B{bc_id}(Final)", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Combine frames
            tac_frame = cv2.resize(tac_frame, (tac_w, max_h))
            bc_frame = cv2.resize(bc_frame, (bc_w, max_h))
            combined = np.zeros((max_h, tac_w + bc_w, 3), dtype=np.uint8)
            combined[:, :tac_w] = tac_frame
            combined[:, tac_w:] = bc_frame
            
            # Add labels
            cv2.putText(combined, "Tacticam (Before/After)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Broadcast (Before/After)", (tac_w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(combined)
        
        out.release()
        logger.info("Before/after comparison video saved to %s", output_path)
    
    def generate_mapping_quality_visualization(self, heatmap_path: str = os.path.join(BASE_DIR, 'mapping_heatmap.png')):
        """Generate heatmap and score chart for mapping quality"""
        if not self.player_mappings:
            logger.warning("Cannot generate visualizations: no mappings available")
            return
        
        logger.info("Generating mapping quality visualizations...")
        
        # Calculate similarity matrix
        similarities = np.zeros((len(self.tacticam_tracks), len(self.broadcast_tracks)))
        for i, tac_track in enumerate(self.tacticam_tracks):
            for j, bc_track in enumerate(self.broadcast_tracks):
                similarities[i, j] = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarities, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'B{t.id}' for t in self.broadcast_tracks],
                   yticklabels=[f'T{t.id}' for t in self.tacticam_tracks])
        plt.title('Player Mapping Similarity Heatmap')
        plt.xlabel('Broadcast Tracks')
        plt.ylabel('Tacticam Tracks')
        plt.savefig(heatmap_path)
        plt.close()
        logger.info("Similarity heatmap saved to %s", heatmap_path)
        
        # Create score chart
        mapping_scores = []
        labels = []
        for tac_id, bc_id in self.player_mappings.items():
            tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
            bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
            if tac_track and bc_track:
                sim = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
                mapping_scores.append(sim)
                labels.append(f'T{tac_id}->B{bc_id}')
        
        if mapping_scores:
            chart_data = {
                'type': 'bar',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Mapping Confidence',
                        'data': mapping_scores,
                        'backgroundColor': ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF'],
                        'borderColor': ['#2A7ABF', '#D54F6A', '#D4A017', '#3A9A9A', '#7A4FD6'],
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'max': 1,
                            'title': {
                                'display': True,
                                'text': 'Similarity Score'
                            }
                        },
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Player Mappings'
                            }
                        }
                    },
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Mapping Confidence Scores'
                        }
                    }
                }
            }
            logger.info("Mapping confidence score chart generated: %s", json.dumps(chart_data, indent=2))
    
    def map_players_advanced(self, similarity_threshold: float = 0.4, secondary_threshold: float = 0.3) -> Dict[int, int]:
        """Advanced player mapping with visualization calls"""
        logger.info("Processing videos...")
        
        self.broadcast_detections = self.process_video_enhanced(
            os.path.join(BASE_DIR, "broadcast.mp4"), 'broadcast'
        )
        self.tacticam_detections = self.process_video_enhanced(
            os.path.join(BASE_DIR, "tacticam.mp4"), 'tacticam'
        )
        
        logger.info("Tracking players...")
        
        self.broadcast_tracks = self.track_players_enhanced(self.broadcast_detections)
        self.tacticam_tracks = self.track_players_enhanced(self.tacticam_detections)
        
        logger.info("Broadcast tracks: %d", len(self.broadcast_tracks))
        logger.info("Tacticam tracks: %d", len(self.tacticam_tracks))
        
        if not self.broadcast_tracks or not self.tacticam_tracks:
            logger.warning("Insufficient tracks for mapping")
            return {}
        
        logger.info("Calculating advanced similarities (primary threshold: %.2f)...", similarity_threshold)
        
        similarities = np.zeros((len(self.tacticam_tracks), len(self.broadcast_tracks)))
        feature_sims = np.zeros((len(self.tacticam_tracks), len(self.broadcast_tracks)))
        
        for i, tac_track in enumerate(self.tacticam_tracks):
            for j, bc_track in enumerate(self.broadcast_tracks):
                feat_sim = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
                feature_sims[i, j] = feat_sim
                
                tac_frames = {det.frame_idx for det in tac_track.detections}
                bc_frames = {det.frame_idx for det in bc_track.detections}
                
                if tac_frames and bc_frames:
                    overlap = len(tac_frames.intersection(bc_frames))
                    total_frames = len(tac_frames.union(bc_frames))
                    temporal_bonus = overlap / total_frames if total_frames > 0 else 0
                else:
                    temporal_bonus = 0
                
                tac_quality = len(tac_track.detections) * tac_track.confidence_score
                bc_quality = len(bc_track.detections) * bc_track.confidence_score
                quality_factor = min(tac_quality, bc_quality) / max(tac_quality, bc_quality)
                
                combined_sim = feat_sim * 0.7 + temporal_bonus * 0.2 + quality_factor * 0.1
                similarities[i, j] = combined_sim
                
                if combined_sim > similarity_threshold:
                    logger.debug("Strong match: T%d <-> B%d (feat:%.3f, temp:%.3f, qual:%.3f, total:%.3f)",
                                tac_track.id, bc_track.id, feat_sim, temporal_bonus, quality_factor, combined_sim)
        
        logger.info("Performing optimal assignment...")
        
        cost_matrix = 1 - similarities
        valid_pairs = similarities > similarity_threshold
        cost_matrix[~valid_pairs] = 999
        
        mappings = {}
        used_broadcast = set()
        used_tacticam = set()
        
        if np.any(valid_pairs):
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_indices, col_indices):
                if similarities[i, j] > similarity_threshold:
                    tac_id = self.tacticam_tracks[i].id
                    bc_id = self.broadcast_tracks[j].id
                    mappings[tac_id] = bc_id
                    used_tacticam.add(i)
                    used_broadcast.add(j)
                    logger.info("✓ Final mapping: T%d -> B%d (%.3f)", tac_id, bc_id, similarities[i, j])
        
        unmapped_tacticam = [i for i in range(len(self.tacticam_tracks)) if i not in used_tacticam]
        unmapped_broadcast = [j for j in range(len(self.broadcast_tracks)) if j not in used_broadcast]
        
        if unmapped_tacticam and unmapped_broadcast:
            logger.info("Secondary matching for %d unmapped tacticam players...", len(unmapped_tacticam))
            logger.info("Using relaxed threshold: %.2f", secondary_threshold)
            
            secondary_similarities = similarities[np.ix_(unmapped_tacticam, unmapped_broadcast)]
            secondary_cost = 1 - secondary_similarities
            secondary_valid = secondary_similarities > secondary_threshold
            secondary_cost[~secondary_valid] = 999
            
            if np.any(secondary_valid):
                sec_row_indices, sec_col_indices = linear_sum_assignment(secondary_cost)
                
                for i, j in zip(sec_row_indices, sec_col_indices):
                    if secondary_similarities[i, j] > secondary_threshold:
                        tac_idx = unmapped_tacticam[i]
                        bc_idx = unmapped_broadcast[j]
                        tac_id = self.tacticam_tracks[tac_idx].id
                        bc_id = self.broadcast_tracks[bc_idx].id
                        
                        mappings[tac_id] = bc_id
                        logger.info("✓ Secondary mapping: T%d -> B%d (%.3f)", tac_id, bc_id, secondary_similarities[i, j])
        
        final_unmapped_tac = [t.id for t in self.tacticam_tracks if t.id not in mappings]
        final_unmapped_bc = [t.id for t in self.broadcast_tracks if t.id not in mappings.values()]
        
        if final_unmapped_tac:
            logger.info("Analyzing %d unmapped tacticam players:", len(final_unmapped_tac))
            for tac_id in final_unmapped_tac:
                tac_track = next(t for t in self.tacticam_tracks if t.id == tac_id)
                best_sim = 0
                best_bc_id = None
                
                for bc_track in self.broadcast_tracks:
                    if bc_track.id not in mappings.values():
                        tac_idx = next(i for i, t in enumerate(self.tacticam_tracks) if t.id == tac_id)
                        bc_idx = next(i for i, t in enumerate(self.broadcast_tracks) if t.id == bc_track.id)
                        sim = similarities[tac_idx, bc_idx]
                        if sim > best_sim:
                            best_sim = sim
                            best_bc_id = bc_track.id
                
                logger.info("T%d: %d detections, conf=%.3f, best_match=B%d(%.3f)",
                           tac_id, len(tac_track.detections), tac_track.confidence_score, best_bc_id, best_sim)
        
        if not mappings:
            logger.warning("No valid mappings found above thresholds")
        
        self.player_mappings = mappings
        
        # Generate visualizations
        self.generate_animated_overlays()
        self.generate_before_after_comparison()
        self.generate_mapping_quality_visualization()
        
        return mappings
    
    def generate_mapping_confidence_analysis(self) -> Dict:
        """Analyze mapping confidence and provide insights"""
        if not self.player_mappings:
            logger.warning("No mappings available for confidence analysis")
            return {}
        
        analysis = {
            'high_confidence': [],
            'medium_confidence': [],
            'low_confidence': [],
            'mapping_quality_distribution': {},
            'recommendations': []
        }
        
        similarities = []
        
        for tac_id, bc_id in self.player_mappings.items():
            tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
            bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
            
            if tac_track and bc_track:
                similarity = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
                similarities.append(similarity)
                
                mapping_info = {
                    'tacticam_id': tac_id,
                    'broadcast_id': bc_id,
                    'similarity': float(similarity),
                    'tac_detections': len(tac_track.detections),
                    'bc_detections': len(bc_track.detections),
                    'tac_confidence': float(tac_track.confidence_score),
                    'bc_confidence': float(bc_track.confidence_score)
                }
                
                if similarity > 0.7:
                    analysis['high_confidence'].append(mapping_info)
                elif similarity > 0.5:
                    analysis['medium_confidence'].append(mapping_info)
                else:
                    analysis['low_confidence'].append(mapping_info)
        
        if similarities:
            analysis['mapping_quality_distribution'] = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'median_similarity': float(np.median(similarities))
            }
        
        if len(analysis['low_confidence']) > 0:
            analysis['recommendations'].append(
                f"Review {len(analysis['low_confidence'])} low-confidence mappings"
            )
        
        if len(self.player_mappings) > 0:
            high_conf_ratio = len(analysis['high_confidence']) / len(self.player_mappings)
            if high_conf_ratio > 0.8:
                analysis['recommendations'].append("Excellent mapping quality achieved")
            elif high_conf_ratio > 0.6:
                analysis['recommendations'].append("Good mapping quality")
            else:
                analysis['recommendations'].append("Consider adjusting similarity thresholds")
        
        logger.info("Mapping confidence analysis completed: %s", json.dumps(analysis, indent=2))
        return analysis
    
    def generate_comprehensive_report(self, output_path: str = os.path.join(BASE_DIR, 'player_mappings.json')) -> Dict:
        """Generate detailed analysis report with visualization references"""
        logger.info("Generating comprehensive report...")
        
        broadcast_stats = {
            'total_tracks': len(self.broadcast_tracks),
            'avg_detections_per_track': float(np.mean([len(t.detections) for t in self.broadcast_tracks])) if self.broadcast_tracks else 0.0,
            'avg_confidence': float(np.mean([t.confidence_score for t in self.broadcast_tracks])) if self.broadcast_tracks else 0.0
        }
        
        tacticam_stats = {
            'total_tracks': len(self.tacticam_tracks),
            'avg_detections_per_track': float(np.mean([len(t.detections) for t in self.tacticam_tracks])) if self.tacticam_tracks else 0.0,
            'avg_confidence': float(np.mean([t.confidence_score for t in self.tacticam_tracks])) if self.tacticam_tracks else 0.0
        }
        
        mapping_details = {}
        for tac_id, bc_id in self.player_mappings.items():
            tac_track = next((t for t in self.tacticam_tracks if t.id == tac_id), None)
            bc_track = next((t for t in self.broadcast_tracks if t.id == bc_id), None)
            
            if tac_track and bc_track:
                similarity = self.calculate_feature_similarity(tac_track.avg_features, bc_track.avg_features)
                mapping_details[f"T{tac_id}_to_B{bc_id}"] = {
                    'similarity_score': float(similarity),
                    'tacticam_detections': len(tac_track.detections),
                    'broadcast_detections': len(bc_track.detections),
                    'tacticam_confidence': float(tac_track.confidence_score),
                    'broadcast_confidence': float(bc_track.confidence_score)
                }
        
        confidence_analysis = self.generate_mapping_confidence_analysis()
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'video_metadata': self.video_metadata,
            'player_mappings': self.player_mappings,
            'mapping_details': mapping_details,
            'confidence_analysis': confidence_analysis,
            'statistics': {
                'total_mapped_players': len(self.player_mappings),
                'broadcast_camera': broadcast_stats,
                'tacticam_camera': tacticam_stats,
                'mapping_success_rate': float(len(self.player_mappings) / max(len(self.tacticam_tracks), 1) * 100)
            },
            'unmapped_players': {
                'tacticam': [t.id for t in self.tacticam_tracks if t.id not in self.player_mappings],
                'broadcast': [t.id for t in self.broadcast_tracks if t.id not in self.player_mappings.values()]
            },
            'visualizations': {
                'animated_overlay': 'matching_animation.gif',
                'before_after_comparison': 'before_after_mapping.mp4',
                'mapping_heatmap': 'mapping_heatmap.png'
            }
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info("Comprehensive report saved to %s", output_path)
        except Exception as e:
            logger.error("Error saving report to %s: %s", output_path, e)
        
        return results

def main():
    MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
    BROADCAST_VIDEO = os.path.join(BASE_DIR, "broadcast.mp4")
    TACTICAM_VIDEO = os.path.join(BASE_DIR, "tacticam.mp4")
    
    logger.info("Checking input files...")
    model_candidates = [
        MODEL_PATH,
        os.path.join(BASE_DIR, "yolov11n.pt"),
        os.path.join(BASE_DIR, "yolov11s.pt"),
        os.path.join(BASE_DIR, "runs", "train", "exp", "weights", "best.pt")
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            logger.info("Model found: %s", candidate)
            break
    
    if not model_path:
        logger.error("No model file found!")
        return
    
    if not os.path.exists(BROADCAST_VIDEO):
        logger.error("Broadcast video not found: %s", BROADCAST_VIDEO)
        return
        
    if not os.path.exists(TACTICAM_VIDEO):
        logger.error("Tacticam video not found: %s", TACTICAM_VIDEO)
        return
    
    logger.info("All input files found!")
    
    try:
        mapper = EnhancedPlayerMapper(model_path)
        
        logger.info("\n%s", "="*60)
        logger.info("ENHANCED CROSS-CAMERA PLAYER MAPPING")
        logger.info("="*60)
        
        mappings = mapper.map_players_advanced(similarity_threshold=0.4, secondary_threshold=0.3)
        
        logger.info("\n%s", "="*60)
        logger.info("GENERATING DETAILED RESULTS")
        logger.info("="*60)
        
        results = mapper.generate_comprehensive_report()
        
        logger.info("\n%s", "="*60)
        logger.info("FINAL SUMMARY")
        logger.info("="*60)
        
        stats = results['statistics']
        logger.info("✓ Successfully mapped %d players", stats['total_mapped_players'])
        logger.info("✓ Tacticam tracks: %d", stats['tacticam_camera']['total_tracks'])
        logger.info("✓ Broadcast tracks: %d", stats['broadcast_camera']['total_tracks'])
        logger.info("✓ Mapping success rate: %.1f%%", stats['mapping_success_rate'])
        
        if mappings:
            logger.info("✓ Player ID mappings: %s", mappings)
        
        unmapped = results['unmapped_players']
        if unmapped['tacticam']:
            logger.warning("⚠ Unmapped tacticam players: %s", unmapped['tacticam'])
        if unmapped['broadcast']:
            logger.warning("⚠ Unmapped broadcast players: %s", unmapped['broadcast'])
        
        logger.info("✓ Visualizations generated:")
        logger.info("  - Animated overlay: %s", results['visualizations']['animated_overlay'])
        logger.info("  - Before/after comparison: %s", results['visualizations']['before_after_comparison'])
        logger.info("  - Mapping heatmap: %s", results['visualizations']['mapping_heatmap'])
        
    except Exception as e:
        logger.error("Error in main execution: %s", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()