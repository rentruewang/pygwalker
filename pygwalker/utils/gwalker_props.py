from __future__ import annotations
from typing import NamedTuple, Callable, Type, TYPE_CHECKING, TypeVar, Generic, Dict, List, Any, Optional
from typing_extensions import Literal
from types import ModuleType
from abc import abstractclassmethod
import sys
import json
import importlib
from .fname_encodings import fname_decode, fname_encode
from pygwalker.base import BYTE_LIMIT


class FieldSpec(NamedTuple):
    """Field specification.

    Args:
    - semanticType: '?' | 'nominal' | 'ordinal' | 'temporal' | 'quantitative'. default to '?'.
    - analyticType: '?' | 'dimension' | 'measure'. default to '?'.
    - display_as: str. The field name displayed. None means using the original column name.
    """
    semanticType: Literal['?', 'nominal', 'ordinal', 'temporal', 'quantitative'] = '?'
    analyticType: Literal['?', 'dimension', 'measure'] = '?'
    display_as: str = None

class PandasLike(NamedTuple):
    "Storage for 2 types, DataFrame and Series."
    DataFrame: type
    Series: type

default_field_spec = FieldSpec()

types: List[PandasLike] = []
if TYPE_CHECKING:
    def _find_df_or_series(module: ModuleType, attr: Literal['DataFrame', 'Series']):
        """
        Finding DataFrame or Series from a module.
        e.g. return module.DataFrame if module is pandas
        """
        try:
            df_or_series = getattr(module, attr)
            return df_or_series
        except AttributeError:
            raise ValueError("The given module {} doesn't contain the following class {}".format(module, attr))
        

    def _try_import_to_list(module: str):
        """
        Import the given module and append it to the `types` string.
        Does nothing if the import fails.
        """
        try:
            pd = importlib.import_module(module)
        except ModuleNotFoundError:
            return

        df = _find_df_or_series(pd, 'DataFrame')
        series = _find_df_or_series(pd, 'Series')

        types.append(PandasLike(DataFrame=df, Series=series))
            
    _try_import_to_list('pandas')
    _try_import_to_list('modin.pandas')
    _try_import_to_list('polars')

DataFrame = TypeVar("DataFrame", *[typ.DataFrame for typ in types])
"""
DataFrame can be *.DataFrame, where * is a module.
"""

Series = TypeVar("Series", *[typ.Series for typ in types])
"""
Series can be *.Series, where * is a module.
"""


def register_prop_getter(module: str):
    """
    A decorator to register a module by their sys.modules handle with a builder function.
    """

    def function(wrapped: Callable[[Type], Type[DataFramePropGetter]]):
        """
        A builder function that takes in a (ModuleType) -> Type[DataFramePropGetter]
        and register it to pygwalker so that pygwalker recognizes the type.
        """

        # Using a () -> Type[DataFramePropGetter] s.t. it evaluates lazily.
        __registered_modules[module] = lambda: wrapped(importlib.import_module(module))
        return wrapped

    return function

class DataFramePropGetter(Generic[DataFrame, Series]):
    """DataFrame property getter"""
    Series = TypeVar("Series")

    @abstractclassmethod
    def infer_semantic(cls, df: DataFrame, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def infer_analytic(cls, df: DataFrame, **kwargs):
        raise NotImplementedError

    @abstractclassmethod
    def to_records(cls, df: DataFrame, **kwargs) -> List[Dict[str, Any]]:
        """Convert DataFrame to a list of records"""
        raise NotImplementedError

    @abstractclassmethod
    def to_matrix(cls, df: DataFrame, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractclassmethod
    def escape_fname(cls, df: DataFrame, **kwargs) -> DataFrame:
        """Encode fname to prefent special characters in field name to cause errors"""
        raise NotImplementedError

    @classmethod
    def series(cls, df: DataFrame, i: int, col: str) -> Series:
        return df[col]

    @classmethod
    def infer_prop(
        cls, df: DataFrame, col: str, i=None, field_specs: Optional[Dict[str, FieldSpec]] = None
    ) -> Dict[str, str]:
        """get IMutField

        Returns:
            (IMutField, Dict)
        """
        if field_specs is None:
            field_specs = {}

        s: cls.Series = cls.series(df, i, col)
        orig_fname = cls.decode_fname(s)
        field_spec = field_specs.get(orig_fname, default_field_spec)
        semantic_type = cls.infer_semantic(s) if field_spec.semanticType == '?' else field_spec.semanticType
        # 'quantitative' | 'nominal' | 'ordinal' | 'temporal';
        analytic_type = cls.infer_analytic(s) if field_spec.analyticType == '?' else field_spec.analyticType
        # 'measure' | 'dimension';
        fname = orig_fname if field_spec.display_as is None else field_spec.display_as
        return {
            'fid': col,
            'name': fname,
            'semanticType': semantic_type,
            'analyticType': analytic_type,
        }

    @classmethod
    def raw_fields(cls, df: DataFrame, **kwargs):
        field_specs = kwargs.get('fieldSpecs', {})
        return [
            cls.infer_prop(df, col, i, field_specs)
            for i, col in enumerate(df.columns)
        ]

    @abstractclassmethod
    def limited_sample(cls, df: DataFrame) -> DataFrame:
        """Return the max sample that can be sent to GraphicWalker"""
        raise NotImplementedError

    @classmethod
    def get_props(cls, df: DataFrame, **kwargs):
        """Remove data volume restrictions for non-JUPyter environments.

        Kargs:
            - env: (Literal['Jupyter' | 'Streamlit'], optional): The enviroment using pygwalker from program entry. Default as 'Jupyter'
        """
        if kwargs.get('env') == 'Jupyter':
            df = cls.limited_sample(df)
        df = cls.escape_fname(df, **kwargs)
        props = {
            'dataSource': cls.to_records(df),
            'rawFields': cls.raw_fields(df, **kwargs),
            'hideDataSourceConfig': kwargs.get('hideDataSourceConfig', True),
            'fieldkeyGuard': False,
            'themeKey': 'g2',
            **kwargs,
        }
        return props

    @classmethod
    def decode_fname(cls, s: Series, **kwargs) -> str:
        """Get safe field name from series."""
        raise NotImplementedError

@register_prop_getter('pandas')
@register_prop_getter('modin.pandas')
def _build_pandas_prop_getter(pandas: ModuleType) -> Type[DataFramePropGetter]:
    DataFrame = pandas.DataFrame
    Series = pandas.Series

    class PandasDataFramePropGetter(DataFramePropGetter[DataFrame, Series]):
        @classmethod
        def limited_sample(cls, df: DataFrame) -> DataFrame:
            if len(df)*2 > BYTE_LIMIT:
                df = df.iloc[:BYTE_LIMIT//2]
            return df

        @classmethod
        def infer_semantic(cls, s: Series):
            v_cnt = len(s.value_counts())
            kind = s.dtype.kind
            return 'quantitative' if (kind in 'fcmiu' and v_cnt > 16) else \
                'temporal' if kind in 'M' else \
                'nominal' if kind in 'bOSUV' or v_cnt <= 2 else \
                'ordinal'

        @classmethod
        def infer_analytic(cls, s: Series):
            kind = s.dtype.kind
            return 'measure' if \
                kind in 'fcm' or (kind in 'iu' and len(s.value_counts()) > 16) \
                    else 'dimension'

        @classmethod
        def series(cls, df: DataFrame, i: int, col: str):
            return df.iloc[:, i]

        @classmethod
        def to_records(cls, df: DataFrame):
            df = df.replace({float('nan'): None})
            return df.to_dict(orient='records')

        @classmethod
        def to_matrix(cls, df: DataFrame, **kwargs) -> List[List[Any]]:
            df = df.replace({float('nan'): None})
            return df.to_dict(orient='tight')

        @classmethod
        def escape_fname(cls, df: DataFrame, **kwargs):
            df = df.reset_index()
            df.columns = [f"{col}_{i}" for i, col in enumerate(df.columns)]
            df = df.rename(fname_encode, axis='columns')
            return df

        @classmethod
        def decode_fname(cls, s: Series, **kwargs):
            fname = fname_decode(s.name)
            fname = json.dumps(fname, ensure_ascii=False)[1:-1]
            return fname

    return PandasDataFramePropGetter

@register_prop_getter('polars')
def _build_polars_prop_getter(pl: ModuleType):
    class PolarsDataFramePropGetter(DataFramePropGetter[pl.DataFrame, pl.Series]):
        Series = pl.Series
        @classmethod
        def limited_sample(cls, df: DataFrame) -> DataFrame:
            if len(df)*2 > BYTE_LIMIT:
                df = df.head(BYTE_LIMIT//2)
            return df

        @classmethod
        def infer_semantic(cls, s: pl.Series):
            v_cnt = len(s.value_counts())
            kind = s.dtype
            return 'quantitative' if kind in pl.NUMERIC_DTYPES and v_cnt > 16 else \
                'temporal' if kind in pl.TEMPORAL_DTYPES else \
                'nominal' if kind in [pl.Boolean, pl.Object, pl.Utf8, pl.Categorical, pl.Struct, pl.List] or v_cnt <= 2 else \
                'ordinal'

        @classmethod
        def infer_analytic(cls, s: pl.Series):
            kind = s.dtype
            return 'measure' if kind in pl.FLOAT_DTYPES | pl.DURATION_DTYPES or \
                   (kind in pl.INTEGER_DTYPES and len(s.value_counts()) > 16) else \
                'dimension'

        @classmethod
        def to_records(cls, df: pl.DataFrame, **kwargs) -> List[Dict[str, Any]]:
            df = df.fill_nan(None)
            return df.to_dicts()

        @classmethod
        def to_matrix(cls, df: pl.DataFrame, **kwargs) -> List[Dict[str, Any]]:
            df = df.fill_nan(None)
            dicts = df.to_dicts()
            return {'columns': list(dicts[0].keys()), 'data': [list(d.values()) for d in dicts]}

        @classmethod
        def escape_fname(cls, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
            df = df.rename({i: fname_encode(i) for i in df.columns})
            return df

        @classmethod
        def decode_fname(cls, s: pl.Series, **kwargs):
            fname = fname_decode(s.name)
            fname = json.dumps(fname, ensure_ascii=False)[1:-1]
            return fname

    return PolarsDataFramePropGetter

__classname2method: Dict[str, Type[DataFramePropGetter]] = {}
__registered_modules: Dict[str, Callable[[], Type[DataFramePropGetter]]] = {}

    
def get_prop_getter(df: DataFrame) -> Type[DataFramePropGetter]:
    if type(df) in __classname2method:
        return __classname2method[type(df)]

    for module in __registered_modules.keys():
        if module in sys.modules:
            imported = importlib.import_module(module)
            DataFrame = getattr(imported, 'DataFrame')
            if isinstance(df, DataFrame):
                __classname2method[DataFrame] = __registered_modules[module]()
                return __classname2method[DataFrame]

    return DataFramePropGetter


def get_props(df: DataFrame, **kwargs):
    props = get_prop_getter(df).get_props(df, **kwargs)
    return props
