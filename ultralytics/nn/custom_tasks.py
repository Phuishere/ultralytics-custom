import inspect
from typing import Union, Optional
from ultralytics.nn.modules import custom_block

CUSTOM_BLOCK_NAMES = frozenset(set([
    name 
    for name, member in inspect.getmembers(custom_block, inspect.isclass)
    if member.__module__ == custom_block.__name__
]))

def parse_custom_block_argument(block_name: str, args: list, ch: Optional[list] = None, f: Optional[int] = None) -> dict[str, Union[str, int, float, list]]:
    if block_name in CUSTOM_BLOCK_NAMES:
        # Get input channel
        assert ch and f
        c1 = ch[f]
        if len(args) > 0: # If has input
            c2 = args[0]
        else: # If no input for c2
            c2 = c1
        args = args[1:]
        return {
            "c1": c1,
            "c2": c2,
            "args": args
        }
    else:
        raise Exception(f"--- ERROR: block_name {block_name} is not included in CUSTOM_BLOCKS {CUSTOM_BLOCK_NAMES}. ---")