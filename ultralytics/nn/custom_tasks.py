import inspect
from typing import Union
from ultralytics.nn.modules import custom_block

CUSTOM_BLOCK_NAMES = frozenset(set([
    name 
    for name, member in inspect.getmembers(custom_block, inspect.isclass)
    if member.__module__ == custom_block.__name__
]))

def parse_custom_block_argument(block_name: str, args) -> list[Union[str, int, float]]:
    if block_name in CUSTOM_BLOCK_NAMES:
        return args
    else:
        raise Exception(f"--- ERROR: block_name {block_name} is not included in CUSTOM_BLOCKS {CUSTOM_BLOCK_NAMES}. ---")