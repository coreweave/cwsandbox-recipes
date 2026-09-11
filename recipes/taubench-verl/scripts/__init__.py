"""Marks scripts/ as a regular package.

verl ships its own top-level ``scripts`` package (converter_hf_to_mcore etc.)
into site-packages. Without this file, Python treats our scripts/ as a mere
namespace portion and verl's regular package wins the import, breaking
``from scripts.launch_gpu_sandbox import ...`` wherever verl is installed.
"""
