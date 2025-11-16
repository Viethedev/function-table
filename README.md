# function-table

`function-table` learns a callable mapping from a small table of inputs -> outputs using XGBoost and exposes a simple, immutable `FunctionTable` class.

Installation
------------

pip install function-table

Usage
-----

```python
from function_table import FunctionTable

inputs = [[0], [1], [2], [3], [4]]
outputs = [[0], [1], [4], [9], [16]]  # y = x^2

f = FunctionTable(inputs, outputs)
print(f(2))
print(f([5, 6, 7]))
print(f[1])
```

License
-------

MIT
