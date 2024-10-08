import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Set canvas size
canvas_width, canvas_height = 1280, 720
mask1 = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
drawing_color = (0, 255, 0)  # Default color: Green
pen_size = 5  # Default pen size

# Load and resize logo
logo = cv2.imread('"C:/Users/LEGION/Desktop/desktop/python/airdraw/png-clipart-harry-potter-gryffindor-logo.png"')  # Replace with the path to your logo
logo_height, logo_width = 100, 100  # Desired size of the logo
# logo = cv2.resize(logo, (logo_width, logo_height))

# Create a drawing color palette
def create_color_palette():
    palette = np.zeros((canvas_height, 160, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 255, 255)]
    box_height = palette.shape[0] // len(colors)
    for i, color in enumerate(colors):
        cv2.rectangle(palette, (0, i * box_height), (160, (i + 1) * box_height), color, -1)
        cv2.rectangle(palette, (0, i * box_height), (160, (i + 1) * box_height), (0, 0, 0), 2)  # Border
    return palette, colors

# Create a pen size palette
def create_pen_palette():
    pen_palette = np.zeros((canvas_height, 160, 3), dtype=np.uint8)
    pen_sizes = [2, 5, 10, 15, 20]
    box_height = pen_palette.shape[0] // len(pen_sizes)
    for i, size in enumerate(pen_sizes):
        cv2.rectangle(pen_palette, (0, i * box_height), (160, (i + 1) * box_height), (200, 200, 200), -1)  # Gray box
        cv2.rectangle(pen_palette, (0, i * box_height), (160, (i + 1) * box_height), (0, 0, 0), 2)  # Border
        cv2.putText(pen_palette, str(size), (30, (i + 1) * box_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return pen_palette, pen_sizes

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

color_palette, colors = create_color_palette()
pen_palette, pen_sizes = create_pen_palette()

# Previous fingertip positions for smooth drawing
prev_index_finger_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame could not be captured.")
        break

    # Resize the frame to match the canvas size and flip it for a mirrored view
    frame = cv2.resize(frame, (canvas_width, canvas_height))
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Only use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get index finger tip and middle finger tip
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Convert to pixel coordinates
        index_finger_pos = (int(index_finger_tip.x * canvas_width), int(index_finger_tip.y * canvas_height))
        middle_finger_pos = (int(middle_finger_tip.x * canvas_width), int(middle_finger_tip.y * canvas_height))

        # Check for color selection
        for i, color in enumerate(colors):
            if (index_finger_pos[0] >= 0 and index_finger_pos[0] < 160 and
                index_finger_pos[1] >= i * (canvas_height // len(colors)) and index_finger_pos[1] < (i + 1) * (canvas_height // len(colors))):
                drawing_color = color  # Change drawing color based on hovered color
                # Highlight selected color
                cv2.rectangle(color_palette, (0, i * (canvas_height // len(colors))), (160, (i + 1) * (canvas_height // len(colors))), (255, 255, 0), 2)

        # Check for pen size selection
        for i, size in enumerate(pen_sizes):
            if (index_finger_pos[0] >= canvas_width - 160 and index_finger_pos[0] < canvas_width and
                index_finger_pos[1] >= i * (canvas_height // len(pen_sizes)) and index_finger_pos[1] < (i + 1) * (canvas_height // len(pen_sizes))):
                pen_size = size  # Change pen size based on hovered size
                # Highlight selected size
                cv2.rectangle(pen_palette, (0, i * (canvas_height // len(pen_sizes))), (160, (i + 1) * (canvas_height // len(pen_sizes))), (255, 255, 0), 2)

        # Check if index finger and middle finger are making a "V sign"
        distance = np.sqrt((index_finger_pos[0] - middle_finger_pos[0]) ** 2 + (index_finger_pos[1] - middle_finger_pos[1]) ** 2)
        if distance < 50:  # Adjust the threshold as necessary
            mask1 = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Clear the canvas

        # Draw the line for smooth drawing
        if prev_index_finger_pos is not None:
            cv2.line(mask1, prev_index_finger_pos, index_finger_pos, drawing_color, pen_size)

        # Update previous fingertip position
        prev_index_finger_pos = index_finger_pos

    else:
        # Reset previous position if no hands are detected
        prev_index_finger_pos = None

    # Combine the original frame with the mask
    img = cv2.add(frame, mask1)

    # Draw the palettes on the frame
    img[0:canvas_height, 0:160] = color_palette  # Color palette on the left
    img[0:canvas_height, canvas_width - 160:canvas_width] = pen_palette  # Pen size palette on the right

    # Overlay the logo
    # img[logo_y:logo_y + logo_height, logo_x:logo_x + logo_width] = logo

    # Add diagonal watermark
    watermark_text = "GODRIC"
    font_scale = 10  # Increase font size
    thickness = 15  # Increase thickness
    text_size = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # Get the text position to center it diagonally
    text_x = (canvas_width - text_size[0]) // 2
    text_y = (canvas_height + text_size[1]) // 2

    # Create a transparent overlay for the watermark
    overlay = img.copy()

    # Put the watermark text on the overlay
    cv2.putText(overlay, watermark_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend the overlay with the original image
    img = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 0)

    # Show the combined image
    cv2.imshow("drawGODmode", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
