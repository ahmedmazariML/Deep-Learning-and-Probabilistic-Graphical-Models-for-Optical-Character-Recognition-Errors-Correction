# coding: utf-8

import numpy as np
import skfuzzy as fuzz

class FuzzyFct:
    def __init__(self, abcd):
        self.x_points = np.arange(abcd[0], abcd[3]+1,1)
        self.fct = fuzz.trapmf(self.x_points, abcd)

    def get_proba(self, point):
        if point in self.x_points:
            indice = int(np.where(self.x_points == point)[0])
            return self.fct[indice]

        else:
            return 0