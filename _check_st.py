"""Check for use_container_width compatibility."""
import streamlit as st
import inspect

# Check st.image signature
sig = inspect.signature(st.image)
params = list(sig.parameters.keys())
print(f"Streamlit version: {st.__version__}")
print(f"st.image params: {params}")
has_ucw = "use_container_width" in params
has_ucow = "use_column_width" in params  
print(f"  has use_container_width: {has_ucw}")
print(f"  has use_column_width: {has_ucow}")
