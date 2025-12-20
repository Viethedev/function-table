# function-table

`function-table` learns a callable mapping from a small table of inputs -> outputs using XGBoost and exposes a simple, immutable `FTable` class.

Improvement (v0.2.0)
------------

- Enforce immutability
- Performance improvement
- Error handling improvement
- Additional utilities

Installation
------------

pip install function-table

Usage
-----

```python
from function_table import FTable

inputs = [[0], [1], [2], [3], [4]]
outputs = [[0], [1], [4], [9], [16]]  # y = x^2

# Create Ftable
f = FTable(inputs, outputs) 

# Use as a function
print(f(2))
```

Output:

```python
[[3.995828628540039]]
```

License
-------

MIT
