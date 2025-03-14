import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gauss)
    return noisy_image

def add_salt_and_pepper_noise(image, prob=0.01):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(prob * total_pixels)
    num_pepper = int(prob * total_pixels)

    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def remove_noise_gaussian(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def remove_noise_median(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def remove_noise_wiener(image):
    kernel = np.ones((3, 3)) / 9
    return convolve2d(image, kernel, mode='same', boundary='symm')

def plot_images(original, noisy, filtered, title="Image Processing Results"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(noisy, cmap='gray'), plt.title('Noisy Image')
    plt.subplot(1, 3, 3), plt.imshow(filtered, cmap='gray'), plt.title('Filtered Image')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this to a valid image file path in your local system
    image_path = "test.jpg"

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Unable to load image from {image_path}")
    else:
        noisy_image_gaussian = add_gaussian_noise(image)
        noisy_image_salt_pepper = add_salt_and_pepper_noise(image)

        filtered_image_gaussian = remove_noise_gaussian(noisy_image_gaussian)
        filtered_image_median = remove_noise_median(noisy_image_salt_pepper)

        plot_images(image, noisy_image_gaussian, filtered_image_gaussian, "Gaussian Noise Removal")
        plot_images(image, noisy_image_salt_pepper, filtered_image_median, "Salt and Pepper Noise Removal")
