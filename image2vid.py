# (960, 1920, 3)
import cv2
import os

class Img2Vid:
    def __init__(self, output_video_path, fps=9, shape = (960, 1920)):

        self.output_video_path = output_video_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        h,w = shape
        self.out = cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (w, h))

    def add_frame(self, image):
        self.out.write(image)

    def release(self):
        self.out.release()
