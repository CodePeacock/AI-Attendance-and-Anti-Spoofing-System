# AI Attendance System With AntiSpoofing

This project implements an advanced attendance system that leverages face recognition technology along with anti-spoofing measures to ensure the authenticity of the attendance process. It's designed to be a comprehensive solution for educational institutions or corporate environments looking to automate their attendance tracking with added security.

## Getting Started

These instructions will guide you through setting up and running the AI Attendance System With AntiSpoofing on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.10 (The project is tested and implemented with Python 3.10)
- Conda (Recommended for managing Python versions and virtual environments)

### Installation

1. **Clone the repository** to your local machine:

```sh
git clone https://github.com/CodePeacock/AI-Attendance-and-Anti-Spoofing-System.git
```

2. **Navigate to the project directory**:

```sh
cd AI-Attendance-and-Anti-Spoofing-System
```

3. **Create a virtual environment**:

- On Windows (using Conda):

```sh
conda create -n sams python=3.10
conda activate ams
```

- On Windows (using venv):

```sh
python -m venv venv
venv\Scripts\activate
```

- On Linux or Mac:

```python
python3 -m venv venv
source venv/bin/activate
```

4. **Install the required dependencies**:

```sh
pip install -r requirements.txt
```

### Running the Project

After installing all the dependencies, you can run the project by executing the `attendance_w_antispoofing.py` script:

```sh
python attendance_w_antispoofing.py
```
This will launch the AI Attendance System With AntiSpoofing application.

## Features

1. 😊Face Recognition-Based Attendance: Automatically marks attendance by recognizing faces.

2. 🤖 Anti-Spoofing: Incorporates anti-spoofing measures to prevent fraudulent attendance.

3. 🏬  Embedding Extraction: Extracts and saves face embeddings for accurate recognition.

4. 📒 Attendance Records: Maintains detailed attendance records for further analysis.
Contributing

## Humble Request
Contributions to the AI Attendance System With AntiSpoofing are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
