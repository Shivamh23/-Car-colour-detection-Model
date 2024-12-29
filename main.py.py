import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Canvas
from PIL import Image, ImageTk


# Function to detect cars, colors, and people
def detect_and_count(frame):
    # Load pre-trained Haar cascades for cars and people
    car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars and people
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    people = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Count cars and people
    car_count = len(cars)
    people_count = len(people)

    for (x, y, w, h) in cars:
        # Extract car region
        car_roi = frame[y:y + h, x:x + w]

        # Calculate average color of the car
        avg_color = np.mean(car_roi, axis=(0, 1))

        # Determine the color of the car
        if avg_color[2] > avg_color[0]:  # Red channel > Blue channel
            color = "Blue"  # Assuming red cars are detected as blue
            rectangle_color = (0, 0, 255)  # Red rectangle
        else:
            color = "Other"
            rectangle_color = (255, 0, 0)  # Blue rectangle

        # Draw rectangle around the car
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        cv2.putText(frame, f"{color} Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rectangle_color, 2)

    for (x, y, w, h) in people:
        # Draw rectangle around the person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Display counts
    cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"People: {people_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame, car_count, people_count


# Function to process video and display results
def process_video():
    cap = cv2.VideoCapture(0)  # Open webcam (use a video file path for pre-recorded footage)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection and count
        processed_frame, car_count, people_count = detect_and_count(frame)

        # Display the processed frame
        cv2.imshow("Car Color and People Detection", processed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# GUI: Load Image and Detect
def browse_and_detect():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    # Load image
    image = cv2.imread(file_path)

    # Detect and count in the image
    processed_image, _, _ = detect_and_count(image)

    # Convert the processed image for display in GUI
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    processed_image = Image.fromarray(processed_image)
    processed_image_tk = ImageTk.PhotoImage(processed_image)

    # Display image in GUI
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=processed_image_tk)
    canvas.image = processed_image_tk


# Set up GUI
app = Tk()
app.title("Car Color Detection and Counting")

Label(app, text="Car Color Detection and Counting Model", font=("Arial", 16)).pack(pady=10)

Button(app, text="Browse Image", command=browse_and_detect, width=20).pack(pady=10)
Button(app, text="Start Video Detection", command=process_video, width=20).pack(pady=10)

canvas = Canvas(app, width=640, height=480)
canvas.pack(pady=10)

app.mainloop()
