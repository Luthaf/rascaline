# -*- coding: utf-8 -*-
"""
Implementation of very basis systems for tests
"""

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
            (0, 1, 1.0, (0.0, 0.0, 1.0)),
            (1, 2, 1.0, (0.0, 0.0, 1.0)),
            (2, 3, 1.0, (0.0, 0.0, 1.0)),
        ]

    def pairs_containing(self, center):
        if center == 0:
            return [
                (0, 1, 1.0, (0.0, 0.0, 1.0)),
            ]
        elif center == 1:
            return [
                (0, 1, 1.0, (0.0, 0.0, 1.0)),
                (1, 2, 1.0, (0.0, 0.0, 1.0)),
            ]
        elif center == 2:
            return [
                (1, 2, 1.0, (0.0, 0.0, 1.0)),
                (2, 3, 1.0, (0.0, 0.0, 1.0)),
            ]
        elif center == 3:
            return [
                (2, 3, 1.0, (0.0, 0.0, 1.0)),
            ]
        else:
            raise Exception("got invalid center")


class EmptySystem(SystemBase):
    def size(self):
        return 0

    def species(self):
        return []

    def positions(self):
        return []

    def cell(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def compute_neighbors(self, cutoff):
        return

    def pairs(self):
        return []
