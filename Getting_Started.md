# Getting Started with the Object Dimensions Detector

This guide will walk you through the steps to get the Object Dimensions Detector application running.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python 3.9:** The application is written in Python.
* **pip:** Python package installer (usually comes with Python).
* **Git (optional but recommended):** For cloning the repository.

## Installation

1.  **Clone the Repository (Optional):**

    If the code is hosted on a Git repository (like GitHub), you can clone it using:

    ```bash
    git clone https://github.com/HarshdeepJ/Object_Volume_Detector.git
    cd Object_Volum_Detector
    ```

2.  **Install Dependencies:**

    The application relies on several Python libraries. You can install them using `pip`. Make sure you are in the project directory (where the `requirements.txt` file, if provided, is located)

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Navigate to the Project Directory:**

    Open your terminal or command prompt and navigate to the directory where `app.py` is located.

2.  **Run the Streamlit Application:**

    Use the `streamlit run` command followed by the name of the main Python script:

    ```bash
    streamlit run app.py
    ```

    Replace `app.py` with the actual name of the Python file that contains the Streamlit application code.


## Using the Application

1.  **Upload Images:** Follow the instructions in the application to upload one (front view) or two (front and side view) images.
2.  **Specify Reference Object:** Enter the known real-world width of a reference object present in your image(s) (e.g., the width of an ID card).
3.  **Provide Camera Parameters:** Enter either the focal length of your camera in pixels or the horizontal and vertical field of view (FoV) in degrees. Accurate parameters are crucial for accurate measurements. You might need to look up these specifications for your camera (e.g., smartphone camera).
4.  **Select Reference Object:** Use the interactive drawing tool in the application to draw a bounding box around the reference object in the uploaded image.
5.  **Run Measurement:** Click the button or follow the prompts in the application to initiate the depth estimation and dimension measurement process.
6.  **View Results:** The application will display the detected objects (if any) with their estimated dimensions and distances overlaid on the image(s), along with numerical results.
7.  **Two-View Analysis (Optional):** If you uploaded a second (side view) image, follow the same steps for that image to obtain depth and height measurements, which will then be combined with the front view results.
