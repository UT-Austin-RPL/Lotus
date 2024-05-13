import os
import imageio
import numpy as np


class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def append_image(self, img, idx=0):
        """Directly append an image to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            self.image_buffer[idx].append(img)

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image"):
        """Append a camera observation to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            if not done:
                self.image_buffer[idx].append(obs[camera_name][::-1])
            else:
                if self.last_images[idx] is None:
                    self.last_images[idx] = obs[camera_name][::-1]
                original_image = np.copy(self.last_images[idx])
                blank_image = np.ones_like(original_image) * 128
                blank_image[:, :, 0] = 0
                blank_image[:, :, -1] = 0
                transparency = 0.7
                original_image = (
                    original_image * (1 - transparency) + blank_image * transparency
                )

                self.image_buffer[idx].append(original_image.astype(np.uint8))

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_vector_obs(self, obs, dones, camera_name="agentview_image"):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name)

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")

import cv2
class VideoWriter2:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def append_image(self, img, idx=0):
        """Directly append an image to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            self.image_buffer[idx].append(img)

    def _add_text_to_frame(self, frame, text):
        """Add text to the bottom-left corner of a frame."""
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        #color = (255, 255, 255)  # White color
        color = (0, 0, 255)
        cv2.putText(frame, text, (10, frame.shape[0] - 10), font, font_scale, color, font_thickness)
        return frame

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image", skill_id=None):
        """Append a camera observation to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            frame = obs[camera_name][::-1]
            if skill_id is not None:
                frame = self._add_text_to_frame(frame, f"skill id = {skill_id}")
            if not done:
                self.image_buffer[idx].append(frame)
            else:
                if self.last_images[idx] is None:
                    self.last_images[idx] = frame
                original_image = np.copy(self.last_images[idx])
                blank_image = np.ones_like(original_image) * 128
                blank_image[:, :, 0] = 0
                blank_image[:, :, -1] = 0
                transparency = 0.7
                original_image = (
                    original_image * (1 - transparency) + blank_image * transparency
                )

                self.image_buffer[idx].append(original_image.astype(np.uint8))

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_vector_obs(self, obs, dones, camera_name="agentview_image", skill_id=None):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name, skill_id=skill_id)


    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")

