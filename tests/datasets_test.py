import sys
sys.path.append("..")

import unittest
import inspect

class TestDatasets(unittest.TestCase):

    def test_splits(self):
        """Test if all datasets can be split into train/val/test phases."""
        # Get all dataset classes from the datasets module
        import datasets

        # Find all classes that inherit from torch.utils.data.Dataset
        dataset_classes = []
        for name, obj in inspect.getmembers(datasets):
            if (inspect.isclass(obj) and
                hasattr(obj, '__bases__') and
                any('Dataset' in str(base) for base in obj.__bases__)):
                dataset_classes.append((name, obj))

        print(f"Found {len(dataset_classes)} dataset classes:")

        # Traverse through each dataset class to check splits
        for class_name, dataset_class in dataset_classes:
            try:
                dataset_class(phase='train')
                dataset_class(phase='val')
                dataset_class(phase='test')
                print(f"  {class_name}: Success")
            except Exception as e:
                self.fail(f"Failed to split dataset into train/val/test. Make sure to implement all phases correctly. Output: {e}")
