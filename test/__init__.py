"""Research scripts imported by tests/ as the ``test`` package.

This file makes ``test`` a regular package.  Without it the directory is only a
namespace package, and any regular ``test`` package on sys.path wins over it,
such as the stdlib one a distro Python ships (Ubuntu's /usr/lib/python3.x/test).
``from test import ...`` in tests/ then fails collection.  The tests put the
repo root first on sys.path, so this package takes precedence.
"""
