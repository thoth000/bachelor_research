import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
import pickle
from scipy.ndimage import distance_transform_edt, gaussian_filter

class VesselnessFilter:
    def __init__(self, scales=[2, 4, 8, 16, 32, 64, 1], beta=5.0, c=0.005, a=0.5, b=0.2):
        self.scales = scales
        self.beta = beta
        self.c = c
        self.a = a
        self.b = b

    @staticmethod
    def apply_gaussian_smoothing(image, sigma):
        """
        Applies Gaussian smoothing.

        Args:
            image (numpy.ndarray): Input image (H, W) or (H, W, 1).
            sigma (float): Standard deviation for Gaussian kernel.

        Returns:
            numpy.ndarray: Smoothed image with the same shape as input.
        """
        if image.ndim == 3 and image.shape[2] == 1:  # Grayscale in [H, W, 1]
            image = image[:, :, 0]  # Convert to [H, W]

        smoothed = cv2.GaussianBlur(image, (0, 0), sigma)

        if smoothed.ndim == 2:  # Convert back to [H, W, 1] if necessary
            smoothed = smoothed[:, :, np.newaxis]

        return smoothed

    @staticmethod
    def compute_hessian(image, sigma):
        """
        Computes the Hessian matrix components.

        Args:
            image (numpy.ndarray): Input image (H, W) or (H, W, 1).
            sigma (float): Scale for Gaussian derivative.

        Returns:
            tuple: Hessian components (Ixx, Iyy, Ixy) with the same shape as input.
        """
        if image.ndim == 3 and image.shape[2] == 1:  # Grayscale in [H, W, 1]
            image = image[:, :, 0]  # Convert to [H, W]

        Ixx = gaussian_filter(image, sigma=sigma, order=(2, 0))
        Iyy = gaussian_filter(image, sigma=sigma, order=(0, 2))
        Ixy = gaussian_filter(image, sigma=sigma, order=(1, 1))

        if Ixx.ndim == 2:  # Convert back to [H, W, 1] if necessary
            Ixx = Ixx[:, :, np.newaxis]
            Iyy = Iyy[:, :, np.newaxis]
            Ixy = Ixy[:, :, np.newaxis]

        return Ixx, Iyy, Ixy

    @staticmethod
    def eigenvalues_of_hessian(Ixx, Iyy, Ixy):
        """
        Computes eigenvalues of the Hessian matrix.

        Args:
            Ixx, Iyy, Ixy (numpy.ndarray): Hessian components (H, W, 1).

        Returns:
            tuple: Eigenvalues (lambda1, lambda2) with the same shape as input.
        """
        trace = Ixx + Iyy
        determinant = Ixx * Iyy - Ixy**2
        discriminant = np.sqrt(np.maximum((trace / 2)**2 - determinant, 0))

        lambda1 = trace / 2 + discriminant
        lambda2 = trace / 2 - discriminant

        return lambda1, lambda2

    def vesselness_filter(self, image):
        """
        Applies the vesselness filter over multiple scales.

        Args:
            image (numpy.ndarray): Input image (H, W, 1).

        Returns:
            numpy.ndarray: Vesselness map (H, W, 1).
        """

        vesselness = np.zeros_like(image, dtype=np.float32)

        for sigma in self.scales:
            smoothed = self.apply_gaussian_smoothing(image, sigma)
            Ixx, Iyy, Ixy = self.compute_hessian(smoothed, sigma)
            lambda1, lambda2 = self.eigenvalues_of_hessian(Ixx, Iyy, Ixy)

            lambda1, lambda2 = np.maximum(lambda1, lambda2), np.minimum(lambda1, lambda2)

            Rb = np.abs(lambda1) / np.sqrt(np.abs(lambda1 * lambda2) + 1e-6)
            Rb = np.clip(Rb, None, 10)
            S = np.sqrt(lambda1**2 + lambda2**2)
            S = np.clip(S, None, 0.01)
            
            v = np.exp(-Rb**2 / (2 * self.beta**2)) * (1 - np.exp(-S**2 / (2 * self.c**2)))
            vesselness = np.maximum(vesselness, v)

        return vesselness, Rb, S

    def vessel_region_info(self, vesselness):
        """
        Computes the vessel region information function.

        Args:
            vesselness (numpy.ndarray): Vesselness map (H, W, 1).

        Returns:
            numpy.ndarray: Vessel region information (H, W, 1).
        """
        P = np.zeros_like(vesselness, dtype=np.float32)

        P[vesselness > self.a] = -1
        P[vesselness < self.b] = 1 - vesselness[vesselness < self.b]

        mask_middle = (vesselness >= self.b) & (vesselness <= self.a)
        P[mask_middle] = -vesselness[mask_middle]

        return P

    def __call__(self, sample):
        """
        Applies the vesselness filter and computes the vessel region information function.

        Args:
            sample (dict): Dictionary containing 'transformed_image' (H, W, 1).

        Returns:
            dict: Updated sample with 'P' added.
        """
        image = sample['transformed_image']

        # Apply the vesselness filter
        vesselness_map, Rb, S = self.vesselness_filter(image)
        sample['Rb'] = Rb
        sample['S'] = S
        sample['vesselness_map'] = vesselness_map
        P = self.vessel_region_info(vesselness_map)

        sample['P'] = P
        
        return sample


def process_vesselness(image, scales, beta=2.0, c=5.0, a=0.5, b=0.2):
    """
    Applies VesselnessFilter to an image and visualizes the P map, Rb, and S maps with colorbars.

    Args:
        image (numpy.ndarray): Input image as a NumPy array.
        scales (list): List of scales for vesselness filter.
        beta (float): Parameter for Rb in vesselness filter.
        c (float): Parameter for S in vesselness filter.
        a (float): Upper threshold for P computation.
        b (float): Lower threshold for P computation.

    Returns:
        dict: Processed results containing vesselness map, Rb, S, and P.
    """

    # Prepare input sample
    sample = {'transformed_image': image.astype(np.float32)}

    # Create VesselnessFilter instance
    vessel_filter = VesselnessFilter(scales=scales, beta=beta, c=c, a=a, b=b)

    # Apply filter
    result = vessel_filter(sample)

    # Visualize and save the P map
    plt.figure(figsize=(8, 6))
    plt.title('P Map')
    im = plt.imshow(result['P'].squeeze(), cmap="viridis")
    plt.axis('off')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('P value', rotation=270, labelpad=15)
    plt.savefig('P_map.png')
    plt.close()

    # Visualize and save the Rb map
    plt.figure(figsize=(8, 6))
    plt.title('Rb Map')
    im = plt.imshow(result['Rb'].squeeze(), cmap="magma")
    plt.axis('off')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Rb value', rotation=270, labelpad=15)
    plt.savefig('Rb_map.png')
    plt.close()

    # Visualize and save the S map
    plt.figure(figsize=(8, 6))
    plt.title('S Map')
    im = plt.imshow(result['S'].squeeze(), cmap="plasma")
    plt.axis('off')
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('S value', rotation=270, labelpad=15)
    plt.savefig('S_map.png')
    plt.close()

    return result



# Define parameters and process the image
image_path = '/home/sano/dataset/DRIVE/training_560/img_patch_0.pkl'
with open(image_path, 'rb') as image_file:
    image = pickle.load(image_file)
    image = np.transpose(image, (1, 2, 0))
process_vesselness(image, scales=[2, 4, 8, 16, 32, 64, 1], beta=4.0, c=0.003, a=0.5, b=0.2)