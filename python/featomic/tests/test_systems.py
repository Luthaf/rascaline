from featomic import SystemBase


class SystemForTests(SystemBase):
    def size(self):
        return 4

    def types(self):
        return [1, 1, 8, 8]

    def positions(self):
        return [[0, 0, 10], [0, 0, 1], [0, 0, 2], [0, 0, 3]]

    def cell(self):
        return [[10, 0, 0], [0, 10, 0], [0, 0, 10]]

    def compute_neighbors(self, cutoff):
        return

    def pairs(self):
        return [
            (0, 1, 1.0, (0.0, 0.0, 1.0), (0, 0, 1)),
            (1, 2, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
            (2, 3, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
        ]

    def pairs_containing(self, atom):
        if atom == 0:
            return [
                (0, 1, 1.0, (0.0, 0.0, 1.0), (0, 0, 1)),
            ]
        elif atom == 1:
            return [
                (0, 1, 1.0, (0.0, 0.0, 1.0), (0, 0, 1)),
                (1, 2, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
            ]
        elif atom == 2:
            return [
                (1, 2, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
                (2, 3, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
            ]
        elif atom == 3:
            return [
                (2, 3, 1.0, (0.0, 0.0, 1.0), (0, 0, 0)),
            ]
        else:
            raise Exception("got invalid atom")


class EmptySystem(SystemBase):
    def size(self):
        return 0

    def types(self):
        return []

    def positions(self):
        return []

    def cell(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def compute_neighbors(self, cutoff):
        return

    def pairs(self):
        return []
