import cv2

# -------------------------------------------------
# GStreamer pipeline for Freenova RAW Bayer camera
# -------------------------------------------------
def gstreamer_pipeline(
    device="/dev/video0",
    width=1280,
    height=720,
    framerate=60,
):
    return (
        f"v4l2src device={device} ! "
        f"video/x-bayer,format=bggr10,width={width},height={height},framerate={framerate}/1 ! "
        f"bayer2rgb ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false"
    )

# -----------------------------
# Main logic
# -----------------------------
def show_camera():
    pipeline = gstreamer_pipeline()

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open camera pipeline")
        print("Check Bayer format (bggr10 vs rggb10) if colors look wrong.")
        return

    print("Camera opened successfully. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow("Freenova CSI Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# -----------------------------
if __name__ == "__main__":
    show_camera()