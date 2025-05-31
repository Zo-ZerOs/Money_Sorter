import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access camera")
    exit()

print("Press SPACE to capture image, or ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif key % 256 == 32:  # SPACE pressed
        # Save the image
        cv2.imwrite("real_img/test_img/image.jpg", frame)
        print("Image captured and saved as testing_image.jpg")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
