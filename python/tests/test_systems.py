# -*- coding: utf-8 -*-
from rascaline import SystemBase


class TestSystem(SystemBase):
    def size(self):
        return 4

    def species(self):
        return [1, 1, 8, 8]

    def positions(self):
        return [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]

    def cell(self):
        return [10, 0, 0, 0, 10, 0, 0, 0, 10]

    def compute_neighbors(self, cutoff):
        return

    def pairs(self):
        return [
            (0, 1, (0.0, 0.0, 1.0)),
            (1, 2, (0.0, 0.0, 1.0)),
            (2, 3, (0.0, 0.0, 1.0)),
        ]


class EmptySystem(SystemBase):
    def size(self):
        return 0

    def species(self):
        return []

    def positions(self):
        return []

    def cell(self):
        return []

    def compute_neighbors(self, cutoff):
        return

    def pairs(self):
        return []
