import unittest
import numpy as np
from typing import List
from nnunetv2.training.network_architecture.cond_unet import _expand_int_param


class TestCondUNetHelpers(unittest.TestCase):
    def test_expand_int_param_scalar(self):
        # Test basic scalar expansion
        self.assertEqual(_expand_int_param(3, 4, "test_param"), [3, 3, 3, 3])
        self.assertEqual(_expand_int_param(np.int32(5), 2, "test_param"), [5, 5])
        
    def test_expand_int_param_sequence(self):
        # Test sequence expansion
        self.assertEqual(_expand_int_param([1, 2, 3], 3, "test_param"), [1, 2, 3])
        self.assertEqual(_expand_int_param((4, 5), 2, "test_param"), [4, 5])

    def test_expand_int_param_wrong_length(self):
        # Test length validation
        with self.assertRaisesRegex(ValueError, "test_param must contain exactly 3 values, got 2"):
            _expand_int_param([1, 2], 3, "test_param")

    def test_expand_int_param_min_value_validation(self):
        # Test min_value validation
        with self.assertRaisesRegex(ValueError, "test_param values must be integers >= 0"):
            _expand_int_param(-1, 3, "test_param", min_value=0)
            
        with self.assertRaisesRegex(ValueError, "test_param values must be integers >= 1"):
            _expand_int_param([1, 0, 2], 3, "test_param", min_value=1)

    def test_expand_int_param_type_validation(self):
        # Test type validation (bools, floats, non-integers)
        with self.assertRaisesRegex(ValueError, "test_param values must be integers >= 0"):
            _expand_int_param(True, 3, "test_param", min_value=0)
            
        with self.assertRaisesRegex(ValueError, "test_param values must be integers >= 0"):
            _expand_int_param([1, False, 2], 3, "test_param", min_value=0)
            
        # A float is not iterable and not an integer, so it raises TypeError
        with self.assertRaises(TypeError):
            _expand_int_param(1.5, 3, "test_param", min_value=0)

        # An iterable of invalid type should raise ValueError
        with self.assertRaisesRegex(ValueError, "test_param values must be integers >= 0"):
            _expand_int_param(["a", "b", "c"], 3, "test_param", min_value=0)

