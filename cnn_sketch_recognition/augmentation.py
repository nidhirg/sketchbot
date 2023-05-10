import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random

def augment_image(image, num_augmented_images=30):
    # Convert image to float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    augmented_images = []

    for _ in range(num_augmented_images):
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        if tf.random.stateless_uniform([], seed=seed) > 0.5:
            image = tf.math.subtract(1.0, image)
        # image = tf.image.resize_with_crop_or_pad(image, 29, 29)
        # Make a new seed.
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        # Random crop back to the original size.
        image = tf.image.stateless_random_crop(
            image, size=[28, 28, 3], seed=seed)



        # Generate a unique seed for each transformation
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        #1 Apply random horizontal flip
        flipped_image = tf.image.stateless_random_flip_left_right(image, seed=seed)

        #2 Apply random vertical flip
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        flipped_image = tf.image.stateless_random_flip_up_down(flipped_image, seed=seed)

        #3 Apply random 90-degree rotations
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        rotated_image = tf.image.rot90(flipped_image, k=tf.random.stateless_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=seed))

        #4 Apply random brightness
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        brightness_adjusted_image = tf.image.stateless_random_brightness(rotated_image, max_delta=0.1, seed=seed)

        # Apply random contrast
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        contrast_adjusted_image = tf.image.stateless_random_contrast(brightness_adjusted_image, lower=1.9, upper=2.9, seed=seed)
        # Slice the first three channels (RGB) from the input image
        rgb_image = tf.slice(contrast_adjusted_image, [0, 0, 0], [-1, -1, 3])

        # Apply random saturation
        seed = tf.random.uniform(shape=[2], minval=0, maxval=2**31-1, dtype=tf.int32)
        saturated_image = tf.image.stateless_random_saturation(rgb_image, lower=0.8, upper=1.5, seed=seed)
        hue_adjusted_image = tf.image.stateless_random_hue(saturated_image, max_delta=0.08, seed=seed)

        # Update the seed for the next operation
        seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

        # Stateless random JPEG quality
        jpeg_quality_image = tf.image.stateless_random_jpeg_quality(hue_adjusted_image, min_jpeg_quality=75, max_jpeg_quality=95, seed=seed)

        # Clip pixel values to the range [0, 1]
        final_image = tf.clip_by_value(jpeg_quality_image, 0, 1)

        augmented_images.append(final_image)

    return augmented_images



