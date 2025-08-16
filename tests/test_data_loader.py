import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_data


class TestDataLoader:
    """Test suite for data_loader.py functionality"""
    
    def setup_method(self):
        """Setup test data for each test method"""
        self.test_data_x = np.random.random((100, 10))
        self.test_data_y = np.random.random((100, 2))
        self.combined_data = np.concatenate([self.test_data_x, self.test_data_y], axis=1)
    
    @pytest.mark.unit
    def test_load_npz_data(self):
        """Test loading npz format data"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            np.savez(tmp.name, x=self.test_data_x, y=self.test_data_y)
            tmp.flush()
            
            try:
                x_train, x_test, y_train, y_test, N = load_data(tmp.name)
                
                # Check shapes after transposition
                assert x_train.shape[0] == 10  # features
                assert x_train.shape[1] == 80  # 80% of 100 samples
                assert y_train.shape[0] == 2   # labels
                assert y_train.shape[1] == 80  # 80% of 100 samples
                assert N == 10
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.unit 
    def test_load_csv_data(self):
        """Test loading CSV format data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            # Write header
            tmp.write(','.join([f'x{i}' for i in range(10)] + ['y1', 'y2']) + '\n')
            # Write data
            for row in self.combined_data:
                tmp.write(','.join(map(str, row)) + '\n')
            tmp.flush()
            
            try:
                x_train, x_test, y_train, y_test, N = load_data(tmp.name, N=10)
                
                assert x_train.shape[0] == 10
                assert y_train.shape[0] == 2
                assert N == 10
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.unit
    def test_load_csv_missing_n(self):
        """Test that CSV loading fails without N parameter"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write('x1,x2,y1\n1,2,3\n')
            tmp.flush()
            
            try:
                with pytest.raises(ValueError, match="請指定 N"):
                    load_data(tmp.name)
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.unit
    def test_load_npy_data(self):
        """Test loading npy format data"""
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, self.combined_data)
            
            try:
                x_train, x_test, y_train, y_test, N = load_data(tmp.name, N=10)
                
                assert x_train.shape[0] == 10
                assert y_train.shape[0] == 2
                assert N == 10
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.unit
    def test_unsupported_format(self):
        """Test that unsupported file formats raise error"""
        with pytest.raises(ValueError, match="只支援"):
            load_data("test.txt")
    
    @pytest.mark.unit
    def test_graph_data_npz(self):
        """Test loading graph data from npz"""
        adjacency_matrix = np.random.randint(0, 2, (5, 5))
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            np.savez(tmp.name, adjacency=adjacency_matrix)
            
            try:
                graph_data = load_data(tmp.name, graph_key="adjacency")
                
                assert isinstance(graph_data, np.ndarray)
                assert graph_data.shape == (5, 5)
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.unit
    def test_graph_data_unsupported_format(self):
        """Test that graph data only supports npz/h5"""
        with pytest.raises(ValueError, match="圖結構資料目前只支援"):
            load_data("test.csv", graph_key="adjacency")