# Audio Playlist Clustering

This project provides two approaches for clustering audio files into playlists 
using machine learning. It uses `.mp3` or `.wav` files from a specified directory 
and groups them into `n` clusters (playlists).

---

## Project Structure
    ├── playlists_v1.json # Output from version 1 (Librosa + KMeans)
    ├── playlists_v2.json # Output from version 2 (ResNet + KMeans) 
    ├── main_v1.py # Version 1: MFCC + Tempo (Librosa) 
    ├── main_v2.py.py # Version 2: Deep Features from ResNet18 
    ├── README.md # This file 
    ├── requirements.txt # Required dependencies 
    └── audio/ # Folder containing your audio files


---

## Installation

1. **Clone the repository:**
2. **(Optional) Create a virtual environment:**
   ```bash
    python -m venv venv
    source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
    pip install -r requirements.txt
   ```
4. Add audio files to the `audio/` directory. 
The audio files should be in `.mp3` or `.wav` format.


## Usage
1. **Version 1: Classic Features (MFCC + Tempo)**
   ```bash
    python main_v1.py --path audio --n 3
   ```
2. **Version 2: Deep Audio Features (ResNet18)**
   ```bash
    python main_v2.py --path audio --n 3
   ```
   
## Output Format
- Each version saves results in the following format:
```json
{
  "playlists": [
    {
      "id": 0,
      "songs": [
        "0fa4cfa4-14d0-4850-88b0-8d21382edadb.mp3",
        ...
      ]
    },
    {
      "id": 1,
      "songs": [
        "456a0433-9e97-400c-987c-f633a8a8f3ff.mp3",
        ...
      ]
    },
    {
      "id": 2,
      "songs": [
        "356a199c-513f-4656-a0b0-f12e2610bfee.mp3",
        ...
      ]
    }
  ]
}
```
