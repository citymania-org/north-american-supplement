import os
from datetime import date

import grf

import lib


DEBUG_DIR = 'debug'

os.makedirs(DEBUG_DIR, exist_ok=True)

g = grf.NewGRF(
    grfid=b'NASS',
    name='North American Houses',
    description='TODO',
)

House = g.bind(lib.House)

House(
    id=1553,
    model=lib.ObjFile('models/test1.obj', light_noise=.1),
    # model=lib.ObjFile('models/1553.obj', noise=(0, 1, 1.5)),
).debug_sprites()

# House(
#     id=1553,
#     path='models/1553.obj',
# )

g.write('nas.grf')
