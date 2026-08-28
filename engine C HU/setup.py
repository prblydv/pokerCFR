from pathlib import Path

from setuptools import Extension, setup
import pybind11


ROOT = Path(__file__).resolve().parent

compile_args = ["/O2", "/std:c++20", "/EHsc"]

setup(
    name="heads-up-native-engine",
    version="0.1.0",
    ext_modules=[
        Extension(
            "heads_up_native_engine",
            [str(ROOT / "heads_up_native_engine.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
)
