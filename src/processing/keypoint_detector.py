"""
src/processing/keypoint_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Keypoint detection supporting SIFT and ORB backends.
Handles grayscale conversion and keypoint count capping internally.
"""

from __future__ import annotations

import cv2
import numpy as np


class KeypointDetector:
    """
    Detects keypoints and computes descriptors for a single image tile.

    Parameters
    ----------
    method : str
        Detection algorithm — ``"SIFT"`` (default) or ``"ORB"``.
    max_keypoints : int
        Maximum number of keypoints to retain per image.

    Raises
    ------
    ValueError
        If an unsupported ``method`` is requested.
    """

    SUPPORTED = ("SIFT", "ORB")

    def __init__(self, method: str = "SIFT", max_keypoints: int = 5000) -> None:
        if method not in self.SUPPORTED:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED}.")
        self.method = method
        self.max_keypoints = max_keypoints
        self._detector = self._build_detector()

    def _build_detector(self) -> cv2.Feature2D:
        """Instantiate the OpenCV detector object."""
        if self.method == "SIFT":
            return cv2.SIFT_create(nfeatures=self.max_keypoints)
        return cv2.ORB_create(nfeatures=self.max_keypoints)

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image to uint8 grayscale.

        Handles (H, W), (H, W, C), and (C, H, W) array layouts.
        """
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[0] == 1:
            gray = image[0]
        elif image.ndim == 3 and image.shape[0] in (3, 4):
            gray = cv2.cvtColor(np.moveaxis(image, 0, -1).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        else:
            gray = image

        # Normalise to uint8 if float or 16-bit
        if gray.dtype != np.uint8:
            lo, hi = gray.min(), gray.max()
            if hi > lo:
                gray = ((gray - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)

        return gray

    def detect(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        Detect keypoints and compute descriptors.

        Parameters
        ----------
        image : np.ndarray
            Input image in any supported layout (see ``_to_gray``).

        Returns
        -------
        keypoints : list[cv2.KeyPoint]
            Detected keypoints sorted by response (strongest first).
        descriptors : np.ndarray
            Corresponding descriptor matrix, shape ``(N, D)``.

        Raises
        ------
        ValueError
            If the image is empty or produces no descriptors.
        """
        if image is None or image.size == 0:
            raise ValueError("Received empty image array.")

        gray = self._to_gray(image)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            return [], np.empty((0,), dtype=np.float32)

        # Sort by response and cap
        keypoints, descriptors = zip(
            *sorted(
                zip(keypoints, descriptors),
                key=lambda x: x[0].response,
                reverse=True,
            )
        )
        keypoints = list(keypoints[: self.max_keypoints])
        descriptors = np.array(descriptors[: self.max_keypoints])

        return keypoints, descriptors
