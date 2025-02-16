# 🚗 Nexar Collision Prediction  
Predict car collisions from dashcam footage **before they happen**!

🥉 **3rd Place Solution** to the [Kaggle Nexar Collision Prediction Competition](https://www.kaggle.com/competitions/nexar-collision-prediction).

📄 **Solution write-up**: [Discussion post on Kaggle](https://www.kaggle.com/competitions/nexar-collision-prediction/discussion/578712).


## 📦 Installation

1. Create a Python environment:
   ```bash
   conda create -n nexar python=3.10
   conda activate nexar
   ```
2. Install Poetry (dependency manager):
   ```bash
   pip install poetry
   ```
3. Install the project dependencies:
   ```bash
   poetry install
   ```


## 📁 Data Setup

1. Download the competition dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/nexar-collision-prediction/data).
2. Place the downloaded ZIP file in the `data/raw/` directory:
   ```
   data/
   └── raw/
       └── nexar-collision-prediction.zip
   ```
3. Unzip the file:
   ```bash
   unzip data/raw/nexar-collision-prediction.zip -d data/raw/
   ```


## 🚀 Usage

### 1. Preprocess the data

Run `notebooks/process_data.ipynb`. This step extracts and processes features from the raw dashcam footage. It may take a few hours depending on your hardware.

After preprocessing, the data will be organized under `data/processed/train/` as follows:

```
data/processed/train/
└── 00001/                  # Unique video ID
    ├── flows/              # Optical flow tensors
    │   ├── 00.pt
    │   ├── 01.pt
    │   └── 02.pt
    ├── frames/             # Frame tensors (e.g., RGB frames)
    │   ├── 00.pt
    │   ├── 01.pt
    │   └── 02.pt
    └── masks/              # Frame-level vehicle masks
        ├── 00.pt
        ├── 01.pt
        └── 02.pt
```

Each subfolder contains PyTorch tensors (`.pt`) representing:
- **flows/**: Optical flow data between consecutive frames, shape (2, H, W).
- **frames/**: Processed frame data, shape (3, H, W).
- **masks/**: Binary masks or target labels for each frame, shape (1, H, W).

### 2. Train a model

Train a single model on the processed data using `notebooks/train_single_model.ipynb`. This typically takes 15–30 minutes. Optionally a seed can be set for reproducibility.


## 🛠️ Project Structure

```
.
├── data/
│   └── raw/                        # Raw data from Kaggle
│   └── processed/                  # Preprocessed data
├── logs/                           # Logging WandB and checkpoints
├── notebooks/
│   └── process_data.ipynb          # Data preprocessing pipeline
│   └── train_single_model.ipynb    # Model training script
├── src/                            # Source code (data loading, models, utils, etc.)
├── pyproject.toml                  # Poetry config file
└── README.md                       # This file
```


## 📫 Contact

For questions or feedback, feel free to reach out via the [Kaggle discussion](https://www.kaggle.com/competitions/nexar-collision-prediction/discussion/578712).
