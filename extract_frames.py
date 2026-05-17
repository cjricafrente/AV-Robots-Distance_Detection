import cv2
import os
import argparse

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save the frames.
        frame_rate (int): Extract every nth frame (default: 1, every frame).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_dir", help="Directory to save frames")
    parser.add_argument("--frame_rate", type=int, default=1, help="Extract every nth frame (default: 1)")

    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.frame_rate)