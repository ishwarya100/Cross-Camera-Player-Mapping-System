from flask import Flask, render_template, jsonify
import json
import os
import base64
from io import BytesIO
from PIL import Image
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = r"C:\Users\user\OneDrive\Desktop\Stealthmode_Assignment"
REPORT_PATH = os.path.join(BASE_DIR, "player_mappings.json")  # Updated to match actual file
GIF_PATH = os.path.join(BASE_DIR, "matching_animation.gif")
MP4_PATH = os.path.join(BASE_DIR, "before_after_mapping.mp4")
HEATMAP_PATH = os.path.join(BASE_DIR, "mapping_heatmap.png")

def load_report():
    try:
        if not os.path.exists(REPORT_PATH):
            logger.error("Report file not found: %s", REPORT_PATH)
            return {}
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            report = json.load(f)
            logger.debug("Report loaded successfully: %s", REPORT_PATH)
            logger.debug("Report contents: %s", json.dumps(report, indent=2))
            return report
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in report file: %s, Error: %s", REPORT_PATH, e)
        return {}
    except Exception as e:
        logger.error("Unexpected error loading report: %s", e)
        return {}

def get_base64_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.debug("Image converted to base64: %s", image_path)
            return img_base64
    except FileNotFoundError:
        logger.warning("Image file not found: %s", image_path)
        return ""
    except Exception as e:
        logger.error("Error processing image %s: %s", image_path, e)
        return ""

def get_base64_gif(gif_path):
    try:
        with open(gif_path, 'rb') as f:
            gif_base64 = base64.b64encode(f.read()).decode('utf-8')
            logger.debug("GIF converted to base64: %s", gif_path)
            return gif_base64
    except FileNotFoundError:
        logger.warning("GIF file not found: %s", gif_path)
        return ""
    except Exception as e:
        logger.error("Error processing GIF %s: %s", gif_path, e)
        return ""

def get_base64_video(video_path):
    try:
        with open(video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')
            logger.debug("Video converted to base64: %s", video_path)
            return video_base64
    except FileNotFoundError:
        logger.warning("Video file not found: %s", video_path)
        return ""
    except Exception as e:
        logger.error("Error processing video %s: %s", video_path, e)
        return ""

@app.route('/')
def dashboard():
    report = load_report()
    
    # Prepare chart data with robust validation
    mapping_details = report.get('mapping_details', {})
    labels = []
    scores = []
    
    if not mapping_details:
        logger.warning("No mapping details found in report")
    
    for key, details in mapping_details.items():
        try:
            similarity_score = details.get('similarity_score')
            if isinstance(similarity_score, (int, float)) and not isinstance(similarity_score, bool):
                labels.append(str(key))  # Ensure key is a string
                scores.append(float(similarity_score))
                logger.debug("Processed mapping: %s, similarity: %s", key, similarity_score)
            else:
                logger.warning("Invalid similarity_score for mapping %s: %s", key, similarity_score)
        except Exception as e:
            logger.error("Error processing mapping details for %s: %s", key, e)
            continue
    
    chart_data = {
        'labels': labels if labels else ['No Data'],
        'datasets': [{
            'label': 'Mapping Confidence',
            'data': scores if scores else [0],
            'backgroundColor': ['#36A2EB', '#FF6384', '#FFCE56', '#4BC0C0', '#9966FF'],
            'borderColor': ['#2A7ABF', '#D54F6A', '#D4A017', '#3A9A9A', '#7A4FD6'],
            'borderWidth': 1
        }]
    }
    
    # Serialize chart data to JSON with error handling
    try:
        chart_data_json = json.dumps(chart_data, ensure_ascii=False)
        logger.debug("Chart data serialized successfully: %s", chart_data_json)
    except Exception as e:
        logger.error("Error serializing chart_data: %s", e)
        chart_data_json = json.dumps({
            'labels': ['Error'],
            'datasets': [{
                'label': 'Mapping Confidence',
                'data': [0],
                'backgroundColor': ['#FF6384'],
                'borderColor': ['#D54F6A'],
                'borderWidth': 1
            }]
        })
        logger.debug("Using fallback chart data: %s", chart_data_json)
    
    # Convert visualizations to base64
    heatmap_base64 = get_base64_image(HEATMAP_PATH)
    gif_base64 = get_base64_gif(GIF_PATH)
    video_base64 = get_base64_video(MP4_PATH)
    
    return render_template('dashboard.html', 
                         report=report,
                         chart_data_json=chart_data_json,
                         heatmap_base64=heatmap_base64,
                         gif_base64=gif_base64,
                         video_base64=video_base64)

@app.route('/report.json')
def get_report_json():
    report = load_report()
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=False)