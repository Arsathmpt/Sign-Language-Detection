Sign Language Detection Model
This Streamlit application is designed to recognize a few specific American Sign Language (ASL) gestures in real-time using a webcam. The model's functionality is restricted to a specific time window.

Core Features
Time-Based Operation: The main sign language detection feature is only active between 6:00 PM and 10:00 PM local time. Outside of these hours, the app will display a message indicating it is unavailable.

Real-Time Webcam Feed: The app uses streamlit-webrtc to access the user's webcam and process the video feed live.

Gesture Recognition (Simulated): The application simulates the detection of a few pre-defined ASL gestures ('Hello', 'Thank You', 'I Love You'). This focuses on demonstrating the core logic (time-based access, UI) rather than requiring a complex, trained model.

Modern Dark UI: Features a custom dark theme for a better user experience.

How to Run the Project
Open a Terminal: Launch your command line tool (PowerShell, Command Prompt, etc.).

Activate Virtual Environment: Navigate to the main submission folder and activate the virtual environment:

# Navigate to the main folder
cd path/to/Nullclass Internship Submission

# Activate the environment
.\venv\Scripts\Activate.ps1

Navigate to Project Folder: Move into this project's directory:

cd "Sign Language Detection"

Install Requirements: Install the necessary Python libraries:

pip install -r requirements.txt

Run the App: Start the Streamlit application:

streamlit run "Sign Language Detection.py"

The application will open in your web browser. The webcam feature will be available if the current time is between 6 PM and 10 PM.