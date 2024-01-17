import numpy as np
import matplotlib.pyplot as plt
from buildSCFpyr import build_scf_pyr, reconstruct_scf_pyr
def test_complex_steerable_pyr():
    # Generate a random image
    np.random.seed(42)
    image_size = 256
    random_image = np.random.rand(image_size, image_size)

    # Apply complex steerable pyramid
    pyr, pind, steermtx, harmonics = build_scf_pyr(random_image)

    # Reconstruct the image
    reconstructed_image = reconstruct_scf_pyr(pyr, pind, steermtx)

    # Display the original and reconstructed images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(random_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(np.real(reconstructed_image), cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()

if __name__ == "__main__":
    test_complex_steerable_pyr()
