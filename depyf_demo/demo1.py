from dataclasses import dataclass

import depyf


@dataclass
class Data:
    x: int
    y: float


print("decompile __init__:\n", depyf.decompile(Data.__init__))
print("decompile __eq__:\n", depyf.decompile(Data.__eq__))
