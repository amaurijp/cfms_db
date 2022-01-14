#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
print('FIREFOX CLEANER')

while True:
    time.sleep(100)
    print('closing FIREFOX windows')
    os.system("pkill -f firefox")    
    