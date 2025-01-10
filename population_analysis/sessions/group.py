import glob
import os
import re


class SessionGroup(object):
    def __init__(self, search_directory=None, nwb_session_kwargs={}, session_cls=None, file_ext=".nwb"):
        # self._loaded_sessions = {}
        self._sessions = []
        self.nwb_session_kwargs = nwb_session_kwargs
        self._file_ext = file_ext
        if session_cls is None:
            from population_analysis.sessions.saccadic_modulation import NWBSession
            self._sess_cls = NWBSession
        else:
            self._sess_cls = session_cls

        if search_directory is not None:
            self.find_sessions(search_directory)

    def get_session_date_from_filename(self, filename):
        # TODO put date in NWB and load into memory so fn doesnt matter? or not cuz memloading is slow
        date = re.match(r".*(\d\d\d\d\-\d\d\-\d\d).*", filename).groups()[0]
        return date

    def find_sessions(self, direc):
        found = glob.glob(os.path.join(direc, f"**/*{self._file_ext}"), recursive=True)
        for filename in found:
            self._sessions.append(filename)

    def _get_file_details(self, sess_name):
        split = re.split(r"(\\|/)", sess_name)[::2]
        folder = "/".join(split[:-1])
        filename = split[-1]
        return folder, filename

    def session_iter(self):
        for sess in self._sessions:
            folder, filename = self._get_file_details(sess)
            try:
                sess = self._sess_cls(os.path.join(folder, filename), **self.nwb_session_kwargs)
                yield filename, sess
            except Exception as e:
                yield (filename, e), None

    def session_names_iter(self):
        for sess in self._sessions:
            yield self._get_file_details(sess)

    def len(self):
        return len(self._sessions)
