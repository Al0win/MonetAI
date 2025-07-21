"""
Basic tests for MonetAI package.

These tests ensure that the core functionality works correctly.
"""

import unittest
import sys
from pathlib import Path
import tensorflow as tf

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.monetai.models import CycleGAN
from src.monetai.data import DataLoader
from src.monetai.utils import setup_gpu


class TestCycleGAN(unittest.TestCase):
    """Test CycleGAN model functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CycleGAN(img_height=256, img_width=256)
        self.batch_size = 2
        self.test_input = tf.random.normal([self.batch_size, 256, 256, 3])
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.img_height, 256)
        self.assertEqual(self.model.img_width, 256)
        self.assertEqual(self.model.channels, 3)
        self.assertIsNotNone(self.model.generator_g)
        self.assertIsNotNone(self.model.generator_f)
        self.assertIsNotNone(self.model.discriminator_x)
        self.assertIsNotNone(self.model.discriminator_y)
    
    def test_generator_output_shape(self):
        """Test generator output shapes."""
        # Test Generator G (Photo -> Monet)
        output_g = self.model.generator_g(self.test_input, training=False)
        self.assertEqual(output_g.shape, self.test_input.shape)
        
        # Test Generator F (Monet -> Photo)  
        output_f = self.model.generator_f(self.test_input, training=False)
        self.assertEqual(output_f.shape, self.test_input.shape)
    
    def test_discriminator_output_shape(self):
        """Test discriminator output shapes."""
        # Test Discriminator X
        output_x = self.model.discriminator_x(self.test_input, training=False)
        self.assertEqual(len(output_x.shape), 4)  # Should be 4D tensor
        
        # Test Discriminator Y
        output_y = self.model.discriminator_y(self.test_input, training=False)
        self.assertEqual(len(output_y.shape), 4)  # Should be 4D tensor
    
    def test_loss_functions(self):
        """Test loss function calculations."""
        # Test discriminator loss
        real = tf.ones([self.batch_size, 16, 16, 1])
        fake = tf.zeros([self.batch_size, 16, 16, 1])
        disc_loss = self.model.discriminator_loss(real, fake)
        self.assertIsInstance(disc_loss, tf.Tensor)
        self.assertGreater(disc_loss.numpy(), 0)
        
        # Test generator loss
        generated = tf.random.normal([self.batch_size, 16, 16, 1])
        gen_loss = self.model.generator_loss(generated)
        self.assertIsInstance(gen_loss, tf.Tensor)
        
        # Test cycle loss
        original = tf.random.normal([self.batch_size, 256, 256, 3])
        cycled = tf.random.normal([self.batch_size, 256, 256, 3])
        cycle_loss = self.model.cycle_loss(original, cycled)
        self.assertIsInstance(cycle_loss, tf.Tensor)
        self.assertGreater(cycle_loss.numpy(), 0)
    
    def test_transform_method(self):
        """Test the transform method."""
        transformed = self.model.transform(self.test_input)
        self.assertEqual(transformed.shape, self.test_input.shape)
        # Output should be in [-1, 1] range (approximately)
        self.assertGreaterEqual(tf.reduce_min(transformed).numpy(), -1.5)
        self.assertLessEqual(tf.reduce_max(transformed).numpy(), 1.5)


class TestDataLoader(unittest.TestCase):
    """Test DataLoader functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader(
            img_height=256,
            img_width=256,
            batch_size=2,
            buffer_size=100
        )
        self.test_image = tf.random.normal([256, 256, 3])
    
    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        self.assertEqual(self.data_loader.img_height, 256)
        self.assertEqual(self.data_loader.img_width, 256)
        self.assertEqual(self.data_loader.batch_size, 2)
        self.assertEqual(self.data_loader.buffer_size, 100)
    
    def test_normalize_function(self):
        """Test image normalization."""
        # Input image in [0, 1]
        input_image = tf.random.uniform([256, 256, 3], 0, 1)
        normalized = self.data_loader.normalize(input_image)
        
        # Output should be in [-1, 1]
        self.assertGreaterEqual(tf.reduce_min(normalized).numpy(), -1.0)
        self.assertLessEqual(tf.reduce_max(normalized).numpy(), 1.0)
    
    def test_random_crop(self):
        """Test random cropping."""
        # Create larger image
        large_image = tf.random.normal([300, 300, 3])
        cropped = self.data_loader.random_crop(large_image)
        
        # Should be cropped to target size
        self.assertEqual(cropped.shape, [256, 256, 3])
    
    def test_random_jitter(self):
        """Test random jitter augmentation."""
        jittered = self.data_loader.random_jitter(self.test_image)
        
        # Output should have same shape as input
        self.assertEqual(jittered.shape, self.test_image.shape)
    
    def test_preprocess_functions(self):
        """Test preprocessing functions."""
        dummy_label = tf.constant("test")
        
        # Test training preprocessing
        processed_train = self.data_loader.preprocess_train(self.test_image, dummy_label)
        self.assertEqual(processed_train.shape, self.test_image.shape)
        
        # Test testing preprocessing
        processed_test = self.data_loader.preprocess_test(self.test_image, dummy_label)
        self.assertEqual(processed_test.shape, self.test_image.shape)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_setup_gpu(self):
        """Test GPU setup function."""
        # This should run without error
        try:
            setup_gpu()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"setup_gpu() raised an exception: {e}")
    
    def test_gpu_detection(self):
        """Test GPU detection."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        # This test just ensures the function can be called
        self.assertIsInstance(gpus, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test a complete forward pass through the model."""
        # Initialize components
        model = CycleGAN(img_height=128, img_width=128)  # Smaller for faster testing
        
        # Create test data
        test_photos = tf.random.normal([2, 128, 128, 3])
        test_monet = tf.random.normal([2, 128, 128, 3])
        
        # Forward pass through generators
        fake_monet = model.generator_g(test_photos, training=False)
        fake_photos = model.generator_f(test_monet, training=False)
        
        # Check shapes
        self.assertEqual(fake_monet.shape, test_photos.shape)
        self.assertEqual(fake_photos.shape, test_monet.shape)
        
        # Forward pass through discriminators
        disc_real_x = model.discriminator_x(test_photos, training=False)
        disc_real_y = model.discriminator_y(test_monet, training=False)
        
        # Check discriminator outputs are 4D
        self.assertEqual(len(disc_real_x.shape), 4)
        self.assertEqual(len(disc_real_y.shape), 4)


def run_tests():
    """Run all tests."""
    print("🧪 Running MonetAI Tests")
    print("=" * 30)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCycleGAN))
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestUtils))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
