

# Cross-Camera Player Mapping Dashboard

<img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" alt="Hi" width="100" />





Hi! This is a cute little project I made to track and match players across two videos (Broadcast & Tacticam). I used Python, YOLO (object detection), OpenCV, Flask, and a simple web dashboard to see the results.

Everything runs offline, just on your computer — no internet required!

----------

## Project Structure

```
Stealthmode_Assignment/
│
├── main.py                     → Main file for player detection, tracking, matching
├── app.py                      → Flask app to display dashboard
├── templates/
│   └── dashboard.html          → Simple HTML dashboard (with graph and animations)
├── static/
│   └── matching_animation.gif  → Animation of matched players
├── broadcast.mp4               → Broadcast camera video
├── tacticam.mp4                → Tacticam video
├── best.pt                     → YOLOv11 trained model
└── player_mappings.json        → Output results after mapping

```

----------

## What This Project Does

### 1. Detect Players

-   Uses YOLOv11 to detect players from both videos.
    
-   Filters detections by confidence to avoid errors.
    

### 2. Track Players

-   Uses the Hungarian algorithm to track players across frames.
    
-   Tracks player IDs properly so they don't shuffle every second.
    

### 3. Match Players Across Cameras

-   Matches players from the Broadcast video to the Tacticam video using:
    
    -   Color similarity
        
    -   Size & shape features
        
-   Uses multi-stage matching (Primary and Secondary matching).
    

### 4. Generates Nice Results

-   Saves mappings in a JSON file (`player_mappings.json`)
    
-   Creates a matching animation GIF
    
-   Creates a confidence score chart (shows how accurate the mapping is)
    

### 5. Dashboard UI (Simple Website)

-   Shows:
    
    -   Matching Animation GIF
        
    -   Confidence Bar Chart using ChartJS
        

----------

## How It Looks

* Dashboard View

* Matching Animation

* Bar chart with matching scores

* Animation showing mapping happening

* Simple, clean HTML dashboard

* Fully offline working

----------


## How To Run

### Clone the Project Repository

You can clone this project from GitHub to your system 

`git clone https://github.com/your-username/stealthmode-assignment.git cd stealthmode-assignment` 

> If you don’t have Git, you can directly download the ZIP file and extract it.
----------

### Install Required Python Libraries

Before running the code, install all necessary libraries. Open terminal (CMD) in the project folder and run:

`pip install flask opencv-python ultralytics numpy scipy pillow` 

These libraries help with video processing, object detection, matching logic, and the web dashboard.

----------

### Prepare the Required Files

You can directly use my files for testing (present in the folder):

-   `broadcast.mp4`: Example broadcast video.
    
-   `tacticam.mp4`: Example tacticam video.
    
-   `best.pt`: Pre-trained YOLOv11 model used in this project.
    

Or:

You can use your own files:

-   Download any two sports videos (from YouTube or other sources), rename them as `broadcast.mp4` and `tacticam.mp4`.
    
-   You can use any YOLOv11 `.pt` model (even your custom-trained model) and place it as `best.pt`.
    

Your project folder should look like this:

```
Stealthmode_Assignment/
│
├── main.py
├── app.py
├── templates/
│   └── dashboard.html
├── static/
│   └── matching_animation.gif (generated later)
├── broadcast.mp4
├── tacticam.mp4
├── best.pt
└── player_mappings.json (generated later)
```




----------

### Run Player Detection, Tracking, and Matching

This script detects players in both videos, tracks them frame-wise, compares features, and creates player mappings + animation.

`python main.py` 

What happens:

-   Detects players using YOLO.
    
-   Tracks players using Hungarian algorithm.
    
-   Matches players across both videos based on features.
    
-   Creates:
    
    -   `player_mappings.json` → shows player ID mappings.
        
    -   `matching_animation.gif` → GIF animation of matched players.
        

----------

### Run the Flask Dashboard (to View Results)

To open the dashboard and see everything visually on your browser:

`python app.py` 

After running, open this link in your browser:

`http://127.0.0.1:5000/` 

You will see:

-   Matching animation GIF
    
-   Confidence scores bar chart
----------

## Output 

What It Contains

`player_mappings.json`

- Player mappings between both cameras

`matching_animation.gif`

- Shows matching between players visually

- Dashboard Chart

- Shows confidence scores in bar graph


----------

## Requirements

-   Python 3.8+
    
-   OpenCV
    
-   Ultralytics YOLOv11
    
-   NumPy & SciPy
    
-   Flask (for the dashboard)
    
-   Basic system with GPU (Optional but helps to speed up detection!)
    

----------

## Summary of How Everything Works


| Step     | What Happens?                                     |
|-----------|--------------------------------------------------|
| Step 1   | Detect players using YOLO                         |
| Step 2   | Track players with Hungarian algorithm            |
| Step 3   | Match players across two videos                   |
| Step 4   | Save mappings in JSON and create matching GIF     |
| Step 5   | Show everything on a simple local Flask dashboard |

----------

## Contributing

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, feel free to contribute.

----------
*Made with Python, patience, and a big smile!💛*
