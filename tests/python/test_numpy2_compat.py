"""
NumPy 2.0+ compatibility tests for HPGL
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import os
os.environ['PATH'] = str(Path(__file__).parent.parent.parent / "src" / "geo_bsd") + ';' + os.environ.get('PATH', '')

try:
    from geo_bsd.geo import (
        ContProperty, IndProperty, SugarboxGrid,
        CovarianceModel, covariance, _require_cont_data,
        _requite_ind_data, _create_hpgl_float_array,
        _create_hpgl_ubyte_array
    )
    HPGL_AVAILABLE = True
except ImportError as e:
    HPGL_AVAILABLE = False
    print(f"Warning: Could not import HPGL: {e}")


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL not available")
class TestNumPy2Compatibility:
    """Test NumPy 2.0+ compatibility"""
    
    def test_numpy_version(self):
        """Log NumPy version for testing"""
        print(f"NumPy version: {np.__version__}")
        assert np.__version__ >= "1.24", "NumPy 1.24+ required"
    
    def test_array_creation_float32(self):
        """Test float32 array creation (NumPy 2.0 compatible)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        assert data.dtype == np.float32
        assert data.shape == (5,)
    
    def test_array_creation_uint8(self):
        """Test uint8 array creation (NumPy 2.0 compatible)"""
        data = np.array([0, 1, 2, 3, 4], dtype='uint8')
        assert data.dtype == np.uint8
        assert data.shape == (5,)
    
    def test_fortran_order_array(self):
        """Test Fortran-order array creation"""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype='float32', order='F')
        assert data.flags['F_CONTIGUOUS']
    
    def test_contproperty_numpy2(self):
        """Test ContProperty with NumPy 2.0 arrays"""
        data = np.zeros(100, dtype='float32', order='F')
        mask = np.ones(100, dtype='uint8', order='F')
        
        prop = ContProperty(data, mask)
        assert prop.data.shape == (100,)
        assert prop.mask.shape == (100,)
    
    def test_indproperty_numpy2(self):
        """Test IndProperty with NumPy 2.0 arrays"""
        data = np.zeros(100, dtype='uint8', order='F')
        mask = np.ones(100, dtype='uint8', order='F')
        
        prop = IndProperty(data, mask, 3)
        assert prop.data.shape == (100,)
        assert prop.indicator_count == 3
    
    def test_array_reshape_numpy2(self):
        """Test array reshape with NumPy 2.0"""
        data = np.arange(500, dtype='float32')
        reshaped = data.reshape((10, 10, 5), order='F')
        assert reshaped.shape == (10, 10, 5)
    
    def test_array_copy_numpy2(self):
        """Test array copy with NumPy 2.0"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        copied = data.copy('F')
        assert np.array_equal(data, copied)
    
    def test_ctypes_pointer_conversion(self):
        """Test ctypes pointer conversion (NumPy 2.0 compatible)"""
        import ctypes as C
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        ptr = data.ctypes.data_as(C.POINTER(C.c_float))
        assert ptr is not None
    
    def test_require_cont_data(self):
        """Test _require_cont_data with NumPy 2.0"""
        data = np.array([1.0, 2.0, 3.0], dtype='float32')
        result = _require_cont_data(data)
        assert result is not None
        assert result.dtype == np.float32
    
    def test_3d_array_creation(self):
        """Test 3D array creation for grid data"""
        data = np.zeros((10, 10, 5), dtype='float32', order='F')
        mask = np.ones((10, 10, 5), dtype='uint8', order='F')
        
        assert data.shape == (10, 10, 5)
        assert data.flags['F_CONTIGUOUS']
        assert mask.flags['F_CONTIGUOUS']
    
    def test_array_strides_numpy2(self):
        """Test array strides with NumPy 2.0"""
        data = np.zeros((10, 10, 5), dtype='float32', order='F')
        strides = data.strides
        assert len(strides) == 3
        assert strides[0] == 4  # float32 size
    
    def test_masked_array_operations(self):
        """Test operations with masked arrays"""
        data = np.arange(100, dtype='float32')
        mask = np.ones(100, dtype='uint8')
        mask[::10] = 0  # Mask every 10th element
        
        prop = ContProperty(data, mask)
        # Count informed values
        informed_count = np.sum(prop.mask)
        assert informed_count == 90


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
