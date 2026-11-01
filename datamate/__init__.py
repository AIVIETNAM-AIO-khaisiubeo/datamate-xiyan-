"""DataMate-XiYan: multi-generator question answering over Vietnamese shop data."""
import pandas as _pd

try:  # pandas>=3 defaults to pyarrow strings; object dtype keeps behaviour identical to pandas 2.x
    _pd.set_option("future.infer_string", False)
except Exception:  # pragma: no cover
    pass

__version__ = "1.0.0"
