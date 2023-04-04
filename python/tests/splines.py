import unittest

import numpy as np

import rascaline


def sine(n, ell, r):
    return np.sin(r)


def cosine(n, ell, r):
    return np.cos(r)


class TestSplines(unittest.TestCase):
    def test_splines_with_n_spline_points(self):
        cutoff_radius = 8.0

        spline_points = rascaline.generate_splines(
            sine,
            cosine,
            max_radial=12,
            max_angular=9,
            cutoff_radius=cutoff_radius,
            n_spline_points=1234,
        )

        # check that the first spline point is at 0
        self.assertEqual(spline_points[0]["position"], 0.0)

        # check that the last spline point is at the cutoff radius
        self.assertEqual(spline_points[-1]["position"], 8.0)

        # ensure correct length for values representation
        self.assertEqual(len(spline_points[52]["values"]["data"]), (9 + 1) * 12)

        # ensure correct length for derivatives representation
        self.assertEqual(len(spline_points[23]["derivatives"]["data"]), (9 + 1) * 12)

        # check values at r = 0.0
        self.assertTrue(
            np.allclose(
                np.array(spline_points[0]["values"]["data"]), np.zeros((9 + 1) * 12)
            )
        )

        # check derivatives at r = 0.0
        self.assertTrue(
            np.allclose(
                np.array(spline_points[0]["derivatives"]["data"]), np.ones((9 + 1) * 12)
            )
        )

        n_spline_points = len(spline_points)
        random_spline_point = 123
        random_x = random_spline_point * cutoff_radius / (n_spline_points - 1)

        # check value of a random spline point
        self.assertTrue(
            np.allclose(
                np.array(spline_points[random_spline_point]["values"]["data"]),
                np.sin(random_x) * np.ones((9 + 1) * 12),
            )
        )

    def test_splines_with_accuracy(self):
        cutoff_radius = 8.0
        spline_points = rascaline.generate_splines(
            sine,
            cosine,
            max_radial=12,
            max_angular=9,
            cutoff_radius=cutoff_radius,
        )

        # check that the first spline point is at 0
        self.assertEqual(spline_points[0]["position"], 0.0)

        # check that the last spline point is at the cutoff radius
        self.assertEqual(spline_points[-1]["position"], 8.0)

        # ensure correct length for values representation
        self.assertEqual(len(spline_points[52]["values"]["data"]), (9 + 1) * 12)

        # ensure correct length for derivatives representation
        self.assertEqual(len(spline_points[23]["derivatives"]["data"]), (9 + 1) * 12)

        # check values at r = 0.0
        self.assertTrue(
            np.allclose(
                np.array(spline_points[0]["values"]["data"]), np.zeros((9 + 1) * 12)
            )
        )

        # check derivatives at r = 0.0
        self.assertTrue(
            np.allclose(
                np.array(spline_points[0]["derivatives"]["data"]), np.ones((9 + 1) * 12)
            )
        )

        n_spline_points = len(spline_points)
        random_spline_point = 123
        random_x = random_spline_point * cutoff_radius / (n_spline_points - 1)

        # check value of a random spline point
        self.assertTrue(
            np.allclose(
                np.array(spline_points[random_spline_point]["values"]["data"]),
                np.sin(random_x) * np.ones((9 + 1) * 12),
            )
        )


if __name__ == "__main__":
    unittest.main()
