## Data Loading (data_loader.py)

Data loader module `data_loader.py` supports formats such as npz, csv, npy, h5, and graph structures.


### Usage

```python
from data_loader import load_data

# Load npz format
x_train, x_test, y_train, y_test, N = load_data("your_data.npz")

# Load csv format
x_train, x_test, y_train, y_test, N = load_data("your_data.csv", N=100)

# Load graph structure data
graph = load_data("your_graph.npz", graph_key="adjacency")
```