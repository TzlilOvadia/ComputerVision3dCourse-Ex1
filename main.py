# This is a sample Python script.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import uint8


class PinspeckCamera:
    def __init__(self, reference_frame_filename: str, occluder_frame_filename: str):
        self.reference_frame_filename = reference_frame_filename
        self.occluder_frame_filename = occluder_frame_filename
        # Load videos
        self.reference_video = cv2.VideoCapture(self.reference_frame_filename)
        self.occluder_video = cv2.VideoCapture(self.occluder_frame_filename)

        self.reference_frames = []
        self.occluder_frames = []

        self.reference_frame = None
        self.occluder_frame = None
        self.image = None

    def videos_to_frames(self):
        # Loop through frames
        # Extract Background Image
        self.get_background()
        # Extract occlusion frames
        self.get_occlusions_frames()

        self.reference_frames = np.array(self.reference_frames)
        self.reference_frame = np.median(self.reference_frames, axis=0).astype(np.uint8)


    def get_occlusions_frames(self):
        while True:
            # Read next frame
            ret, frame = self.occluder_video.read()
            self.occluder_frames.append(frame)
            # If there are no more frames, exit the loop
            if not ret:
                break

    def get_background(self):
        counter = 0
        while counter < 100:
            # Read next frame
            ret, frame = self.reference_video.read()
            self.reference_frames.append(frame)
            # If there are no more frames, exit the loop
            counter += 1
            if not ret:
                break


    def compute_diff(self):
        tot = []
        c = 1
        for occ in self.occluder_frames:
            dif = np.abs(self.reference_frame - occ)

            alfa, stretched_img = self.process_dif(dif)

            blended = self.blend_to_background(alfa, stretched_img)

            blended = cv2.bilateralFilter(blended,5,75,75)

            tot.append(blended.astype(np.uint8))

            batch = tot[-min(3,c):]
            stacked = np.stack(batch, axis=0)
            t = np.median(stacked, axis=0).astype(np.uint8)

            cv2.imshow("", t)
            cv2.waitKey(1000)
            c+=1

        return tot

    def blend_to_background(self, alfa, stretched_img):
        blended = cv2.addWeighted(self.reference_frame, 1 - alfa, stretched_img, alfa, 0)
        return blended

    def process_dif(self, dif):
        dif = cv2.bilateralFilter(dif, 5, 75, 75)
        alpha, stretched_img = self.linear_stretch(dif)
        np.clip(alpha, 0, 255, out=stretched_img)
        alfa = .6
        return alfa, stretched_img

    def linear_stretch(self, dif):
        mi, mx = np.min(dif), np.max(dif)
        stretched_img = np.zeros(dif.shape, dtype=np.uint8)
        # Stretch the pixel values using linear stretching
        alpha = (dif - mi / (mx - mi)) * 10
        return alpha, stretched_img

    def capture(self):
        self.videos_to_frames()
        return self.compute_diff()

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

if __name__ == '__main__':
    occluder_fp = "block.mp4"
    ref_fp = "ref.mp4"
    camera = PinspeckCamera(reference_frame_filename=ref_fp, occluder_frame_filename=occluder_fp)
    imgs = camera.capture()


    for image in imgs:
        b, g, r = cv2.split(image.astype(uint8))
        # Apply histogram equalization to each channel
        eq_b = cv2.equalizeHist(b)
        eq_g = cv2.equalizeHist(g)
        eq_r = cv2.equalizeHist(r)
        out = cv2.merge((eq_b, eq_g, eq_r))
        plt.imshow(out)
        plt.show()





