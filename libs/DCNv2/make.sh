#!/usr/bin/env bash
python setup.py build develop --install-dir $(python -m site --user-site)
