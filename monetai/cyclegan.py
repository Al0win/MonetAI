


import tensorflow as tf
tf.config.list_physical_devices('GPU')




import tensorflow as tf
import numpy as np


import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

AUTOTUNE = tf.data.AUTOTUNE



import os

# Convert the path to proper format using os.path
tfrecord_path = os.path.abspath(os.path.join('..', 'datasets', 'photo_tfrec', 'trainA', 'photo00-352.tfrec'))

# Now inspect the TFRecord
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

# Take one example and print its raw format
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print("TFRecord structure:")
    print(example.features.feature)


# Define paths to your TFRecord directories
trainA_path = os.path.abspath(os.path.join('..', 'datasets', 'photo_tfrec', 'trainA'))
trainB_path = os.path.abspath(os.path.join('..', 'datasets', 'monet_tfrec', 'trainB'))
testA_path = os.path.abspath(os.path.join('..', 'datasets', 'photo_tfrec', 'testA'))
testB_path = os.path.abspath(os.path.join('..', 'datasets', 'monet_tfrec', 'testB'))

# Function to parse a TFRecord example
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),  # Image stored as raw bytes
        'target': tf.io.FixedLenFeature([], tf.string),  # Target label
        'image_name': tf.io.FixedLenFeature([], tf.string),  # Image name
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    # Decode and preprocess the image
    image = tf.io.decode_jpeg(parsed_features['image'])  # Decode raw JPEG bytes
    image = tf.image.resize(image, [256, 256])  # Resize to 256x256 (adjust as needed)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    label = parsed_features['target']  # Keep the label as string (for now)
    return image, label

# Function to create a dataset from TFRecord files in a directory
def load_tfrecord_dataset(tfrecord_dir):
    tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrec')]
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    parsed_dataset = raw_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    return parsed_dataset

# Load datasets for training and testing
train_photos = load_tfrecord_dataset(trainA_path)
train_monet = load_tfrecord_dataset(trainB_path)
test_photos = load_tfrecord_dataset(testA_path)
test_monet = load_tfrecord_dataset(testB_path)

# Validate the data pipeline by taking one example
for image, label in train_photos.take(1):
    print(f"Image shape: {image.shape}, Label: {label.numpy().decode('utf-8')}")
    plt.figure(figsize=(6, 6))
    plt.imshow(image.numpy())  # Convert the tensor to a NumPy array
    plt.title(f"Label: {label.numpy().decode('utf-8')}")
    plt.axis('off')  # Remove axes for better visualization
    plt.show()
    
for image, label in train_monet.take(1):
    print(f"Image shape: {image.shape}, Label: {label.numpy().decode('utf-8')}")
    plt.figure(figsize=(6, 6))
    plt.imshow(image.numpy())  # Convert the tensor to a NumPy array
    plt.title(f"Label: {label.numpy().decode('utf-8')}")
    plt.axis('off')  # Remove axes for better visualization
    plt.show()


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

for image, label in train_photos.take(1):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    plt.imshow(random_crop(image).numpy())
    plt.axis('off')
    plt.show()



# Normalize the image to [-1, 1]
def normalize(image):
    image = (image * 2) - 1  # Rescale to [-1, 1]
    return image

# Loop through the dataset
for image, label in train_photos.take(1):
    normalized_image = normalize(image)
    # Rescale for visualization to [0, 1]
    rescaled_image = (normalized_image + 1) / 2  # Scale back to [0, 1]
    plt.imshow(normalized_image.numpy())
    plt.axis('off')
    plt.show()

    plt.imshow(rescaled_image.numpy())
    plt.axis('off')
    plt.show()


def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

for image, label in train_photos.take(1):
    jitterised_image = random_jitter(image)
    # Rescale for visualization to [0, 1]
    plt.imshow(jitterised_image.numpy())
    plt.axis('off')
    plt.show()

    plt.imshow(image.numpy())
    plt.axis('off')
    plt.show()


def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

for image, label in train_photos.take(1):
    preprocessed_image= preprocess_image_train(image, label)
    # Rescale for visualization to [0, 1]
    plt.imshow(preprocessed_image.numpy())
    plt.axis('off')
    plt.show()

    plt.imshow(image.numpy())
    plt.axis('off')
    plt.show()


def preprocess_image_test(image, label):
  image = normalize(image)
  return image


train_photos = train_photos.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_monet = train_monet.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_photos = test_photos.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_monet = test_monet.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)


sample_photo = next(iter(train_photos))
sample_monet = next(iter(train_monet))


plt.subplot(121)
plt.title('Photo')
plt.imshow(sample_photo[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Photo with random jitter')
plt.imshow(random_jitter(sample_photo[0]) * 0.5 + 0.5)


plt.subplot(121)
plt.title('Monet')
plt.imshow(sample_monet[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet with random jitter')
plt.imshow(random_jitter(sample_monet[0]) * 0.5 + 0.5)



OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


to_monet = generator_g(sample_photo)
to_photo = generator_f(sample_monet)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_photo, to_monet, sample_monet, to_photo]
title = ['Photo', 'To Monet', 'Monet', 'To Photo']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()


plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real Monet painting?')
plt.imshow(discriminator_y(sample_monet)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real Photo?')
plt.imshow(discriminator_x(sample_photo)[0, ..., -1], cmap='RdBu_r')

plt.show()



LAMBDA = 10


loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5


def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1


# As shown above, generator $G$ is responsible for translating image $X$ to image $Y$. Identity loss says that, if you fed image $Y$ to generator $G$, it should yield the real image $Y$ or something close to image $Y$.
# 
# If you run the monet-to-photo model on a photo or the photo-to-monet model on a monet, it should not modify the image much since the image already contains the target class.
# 
# $$Identity\ loss = |G(Y) - Y| + |F(X) - X|$$

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


# Initialize the optimizers for all the generators and the discriminators.

generator_g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


# ## Checkpoints

# Create checkpoint directory if it doesn't exist
checkpoint_path = "./checkpoints/train"
os.makedirs(checkpoint_path, exist_ok=True)

# Set up checkpointing
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                          generator_f=generator_f,
                          discriminator_x=discriminator_x,
                          discriminator_y=discriminator_y,
                          generator_g_optimizer=generator_g_optimizer,
                          generator_f_optimizer=generator_f_optimizer,
                          discriminator_x_optimizer=discriminator_x_optimizer,
                          discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restore the latest checkpoint if it exists
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored from:', ckpt_manager.latest_checkpoint)
else:
    print('Initializing from scratch.')


# ## Training
# 
# Note: This example model is trained for fewer epochs than the paper (200) to keep training time reasonable.

EPOCHS = 50


def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


# Even though the training loop looks complicated, it consists of four basic steps:
# 
# * Get the predictions.
# * Calculate the loss.
# * Calculate the gradients using backpropagation.
# * Apply the gradients to the optimizer.

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


# Training loop
for epoch in range(EPOCHS):
    start = time.time()
    
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_photos, train_monet)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print('.', end='')
        n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_photo) so that the progress of the model
    # is clearly visible.
    generate_images(generator_g, sample_photo)

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                          ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))


# ## Generate using test dataset

# Run the trained model on the test dataset
for inp in test_photos.take(5):
  generate_images(generator_g, inp)


import zipfile
# Ensure output directory exists
output_dir = "photo_to_monet"
zip_filename = "images.zip"
os.makedirs(output_dir, exist_ok=True)

# Function to clear the output directory
def clear_output_directory(output_dir):
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    print(f"Cleared {output_dir}.")

# Function to save images
def save_images(dataset, generator, output_dir):
    image_count = 0
    for photo_batch in dataset:
        generated_images = generator(photo_batch, training=False)
        for img in generated_images:
            img = (img + 1) / 2  # Rescale to [0, 1] for saving
            output_path = os.path.join(output_dir, f"monet_{image_count:04d}.jpg")
            tf.keras.preprocessing.image.save_img(output_path, img.numpy())
            image_count += 1
    print(f"Generated {image_count} images in {output_dir}.")

# Function to zip generated images
def zip_generated_images(output_dir, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    print(f"Zipped images saved to {zip_filename}.")
    
# Clear the output directory before generating images
clear_output_directory(output_dir)

# Generate images and save them
save_images(test_photos, generator_g, output_dir)

# Zip the saved images for submission
zip_generated_images(output_dir, zip_filename)


get_ipython().system('python -m pytorch_fid ../datasets/monet_jpg/testB photo_to_monet')

