import glob
import os
import re

from population_analysis.sessions.saccadic_modulation import NWBSession


class NWBSessionGroup(object):
    def __init__(self, search_directory=None):
        # self._loaded_sessions = {}
        self._sessions = []
        if search_directory is not None:
            self.find_sessions(search_directory)

    def get_session_date_from_filename(self, filename):
        # TODO put date in NWB and load into memory so fn doesnt matter? or not cuz memloading is slow
        date = re.match(r".*(\d\d\d\d\-\d\d\-\d\d).*", filename).groups()[0]
        return date

    def find_sessions(self, direc):
        for filename in glob.glob(direc + "/**/*.nwb", recursive=True):
            self._sessions.append(filename)

    def _get_file_details(self, sess_name):
        split = re.split(r"(\\|/)", sess_name)[::2]
        folder = "/".join(split[:-1])
        filename = split[-1][:-len(".nwb")]
        return folder, filename

    def session_iter(self):
        for sess in self._sessions:
            folder, filename = self._get_file_details(sess)
            yield filename, NWBSession(folder, filename)

    def session_names_iter(self):
        for sess in self._sessions:
            yield self._get_file_details(sess)

    def len(self):
        return len(self._sessions)
