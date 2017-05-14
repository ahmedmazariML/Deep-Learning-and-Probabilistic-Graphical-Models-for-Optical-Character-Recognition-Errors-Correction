# coding: utf-8
from colorama import Fore, Style


class WordFeature:
    def __init__(self):
        """
        Initialize instance with default values
        """
        self.fontsize = 0
        self.category = set()
        self.row = None
        self.col = None
        self.proba_row = 0
        self.proba_col = 0

        self.word = ""
        self.left = list()
        self.right = list()
        self.top = list()
        self.bottom = list()

    def __str__(self):
        label = self.get_label()
        if label != "null" :
            return "{}{}{} -*- {}{}{}".format(
                Fore.CYAN,
                self.word,
                Style.RESET_ALL,
                Fore.MAGENTA, label,
                Style.RESET_ALL)
        else :
            return "{}".format(self.word)

    def __repr__(self):
        return str(self)

    def add_letter(self, char, left, right, top, bottom):
        self.word += char
        self.left.append(left)
        self.right.append(right)
        self.top.append(top)
        self.bottom.append(bottom)
        if len(self.right) != len(self.word):
            print(self.word, 'PROBLEME', len(self.word), len(self.right))

    def get_label(self):
        if self.category:
            label = ".".join(self.category)
        else:
            label = 'null'
        return label

    def add_category(self, category):
        self.category.add(category)

    def remove_category(self, category):
        self.category.remove(category)

    def clean_category(self):
        self.category = set()

    def drop_add_category(self, category):
        self.category = set()
        self.category.add(category)

    def contains_category(self, category):
        return category in self.category

    def change_if_best(self, indice, proba, mode="row"):
        if mode == "row" and proba > self.proba_row:
            self.proba_row = proba
            self.row = indice

        if mode == "col" and proba > self.proba_col:
            self.proba_col = proba
            self.col = indice

    def compute_center_coordinate(self):
        r = max(self.right)
        l = min(self.left)
        b = max(self.bottom)
        t = min(self.top)

        width = r - l
        height = b - t
        return l + (width / 2), t +(height / 2)

    def get_representation(self):
        x, y = self.compute_center_coordinate()
        repr = "{}".format(self.word)

        r = max(self.right)
        l = min(self.left)
        b = max(self.bottom)
        t = min(self.top)

        repr += ";{};{};{};{}".format(t, l, b, r)
        repr += ";{};{};{}".format(x, y, self.fontsize)
        label = self.get_label()
        repr += "\t{}\n".format(label)
        return repr

    def get_coord(self):
        return (min(self.left), max(self.right), min(self.top), max(self.bottom))