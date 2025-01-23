# AlFanar-PPE

A computer vision system to detect **Personal Protective Equipment (PPE)** violations for three classes:

1. **No_Gloves**  
2. **No_Helmet**  
3. **No_Sleeves**

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Setup and Installation](#setup-and-installation)
  - [Using Conda](#using-conda)
  - [Using Virtual Environment](#using-virtual-environment)
  - [Install Required Packages](#install-required-packages)
- [Data Description](#data-description)
- [Inference Code](#inference-code)
- [Train & Tune Code](#Train-Tune-Code)
---

## Repository Structure

Below is an example repository structure.  
If you want to generate an actual tree of your current folder, install **tree** and run it, then paste the output here:

```bash
sudo apt-get install tree    # Install tree if not available
cd /path/to/AlFanar-PPE     # Navigate to your project folder
tree
```
- output
```bash
cv_6/
├── a.env
├── best.pt
├── cameras.streams
├── dashboard.py
├── data.yaml
├── inference.py
├── README.md
├── runs
│   └── detect
├── training.py
├── tune.py
├── train
│   ├── images
│   ├── labels
│   └── labels.cache
├── val
│   ├── images
│   ├── labels
│   └── labels.cache
├── yolov8n.pt
└── yolov8s.pt

```

## Configuration
*List out the specific environment or hardware details needed for this project:*
- **CUDA Version:** <insert CUDA Version: 12.4 >
- **Operating System:** <insert OS version: Ubuntu 24.04.1 LTS>
- **Python Version:** <insert Python 3.8.20 >
- **GPU Model:** <write gpu model: RTX 4070 ti super, VRAM16GB> 

## Setup and Installation

### Using Conda
```sh
conda create -n <env-name> python==<version-number>
conda activate <env-name>
```

### Using Virtual Environment
```sh
python3 -m venv <env-name>
source <env-name>/bin/activate
```

### Install Required Packages
```sh
pip install -r requirements.txt
```

## Data Description

This project utilizes a **10K-image dataset** formatted for YOLO-based object detection. Below is an overview of how the data is structured, processed, and any key considerations to keep in mind when training or fine-tuning the model.

## Inference Code
*Instructions on how to run inference.*

1. **Navigate to the inference directory.**
    ```sh
    cd cv_6
    ```
2. **Execute `inference.py` with parameters:**
    ```sh
    python inference.py 
    ```

      
Also, upload the weights to the [release page](https://github.com/WakebDataScience/template/releases/new) and describe each one uploaded with results, epochs, and parameters used. Then, place the weights in the [*inference/weights/*](inference/weights/) directory after downloading them
<you have to create .gitignore so you don't upload the wieghts to the repository>

## Train & Tune Code
*Instructions on how to train the model, including arguments and their descriptions.*

1. **Navigate to the training directory.**
    ```sh
    cd cv_6
    ```
2. **Run the training script `train.py` with parameters:**
    ```sh
    python training.py 
    ```

1. **Navigate to the Tuning directory.**
    ```sh
    cd cv_6
    ```
2. **Run the training script `tune.py` with parameters:**
    ```sh
    python tune.py 
    ```
