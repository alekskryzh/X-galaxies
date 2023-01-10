#! /usr/bin/env python

"""
Module with functions for writing out results
"""

import time
import shelve
from os import path
from os import remove
import numpy as np


class ResultsFile(object):
    """
    Class for writing out fitted parameters of X-structures
    """
    def __init__(self, file_name, column_names, keep_ordered=True, make_stats=True):
        self.file_name = file_name
        self.keep_ordered = keep_ordered
        self.make_stats = make_stats
        self.column_names = ["radius"] + column_names
        if make_stats:
            self.column_names.extend(["mean", "std"])
        self.write_header()

    def write_header(self):
        fout = open(self.file_name, "w")
        fout.truncate(0)
        fout.write("#")
        for name in self.column_names:
            fout.write("%10s " % name)
        fout.write("\n")
        fout.close()

    def write_data(self, radius, data):
        fout = open(self.file_name, "a")
        fout.write("%11.1f " % radius)
        for value in data:
            fout.write("%10.3f " % value)
        if self.make_stats:
            fout.write("%10.3f" % np.mean(data))
            fout.write("%10.3f\n" % np.std(data))
        else:
            fout.write("\n")
        fout.close()
        self.sort_by_first_column()

    def sort_by_first_column(self):
        data = np.genfromtxt(self.file_name)
        if data.ndim < 2:
            # File doesn't contain at least two lines and can't be sorted.
            return
        inds = np.argsort(data[:, 0]) + 1  # One is added to skip the header
        if all(np.diff(inds) >= 0):
            # File is sorted; nothing to do
            return
        with open(self.file_name) as unsorted_file:
            all_lines = unsorted_file.readlines()
        self.write_header()
        sorted_file = open(self.file_name, "a")
        for line_number in inds:
            sorted_file.write(all_lines[line_number])
        sorted_file.close()

    def get_columns(self, list_of_columns):
        return np.genfromtxt(self.file_name, usecols=list_of_columns, unpack=True)


class LogFile(object):
    """
    Class perpesents a log-file
    """
    def __init__(self, file_name):
        self.file_name = file_name
        fout = open(file_name, "w")
        fout.truncate(0)
        fout.close()

    def write(self, line, show=False):
        """
        Write line to a log file
        """
        current_time = time.localtime()
        time_string = time.strftime('%Y.%m.%d %H:%M:%S ', current_time)
        fout = open(self.file_name, "a")
        line_to_write = "%s: %s\n" % (time_string, line)
        if show:
            print(line_to_write, end='')
        fout.write(line_to_write)
        fout.close()


class ResultShelve(object):
    """
    A wrapper around shelve module for saving sets of fitting objects
    (for example as a function of the ellipse radius).
    """
    def __init__(self, file_name):
        if path.exists(file_name):
            remove(file_name)
        self.file_name = file_name
        # Create empty database with two lists: one for the radius
        # value and the second for objects
        database = shelve.open(file_name)
        database["radius"] = []
        database["objects"] = []
        database.close()

    def append(self, radius, obj):
        """
        Append an object for the given radius
        """
        database = shelve.open(self.file_name)
        radius_list = database["radius"]
        radius_list.append(radius)
        database["radius"] = radius_list
        objects_list = database["objects"]
        objects_list.append(obj)
        database["objects"] = objects_list
        database.close()
