import time
import sqlite3

SETUP_SAMPLE_ID = -1  # value of 'sample' used to identify the 'setup'
                      # phase of CosmoSIS.

# An instance of class Timer is used to collect wall-clock timing
# information, and to organize that information for analysis.
#
# Intended use:
#
#  t = timer.Timer("timer-records.db") # Create a SQLite3 db to store records
#  t.start("module-a", 1)              # start collecting a sample
#  ...                                 # do the work you want timed.
#  t.capture()                         # stop the timer, save the record
#  t.close()                           # flush any remaining records
#
class Timer(object):
    COMMIT_INTERVAL = 100 # commit records after this many inserts

    # Create a Timer object, with data recorded in a SQLite3 database in
    # file 'filename'. Use filename = ":memory:" for an in-memory
    # database.
    def __init__(self, filename):
        self.filename = filename
        self.con = sqlite3.connect(filename)
        self.con.execute("CREATE TABLE IF NOT EXISTS Samples "
                         "(module TEXT, sample INTEGER, t REAL);")
        self.num_samples = 0
        self.t1 = None
        self.current_record = None

    # Start collecting data for the module 'modname', and for sample
    # 'sample'.
    def start(self, modname, sample):
        self.num_samples += 1
        self.current_record = [ modname, sample, time.time() ]

    def capture(self):
        self.current_record[2] = time.time() - self.current_record[2]
        self.con.execute("INSERT INTO Samples (module, sample, t) VALUES (?, ?, ?);",
                         self.current_record)
        if self.num_samples % Timer.COMMIT_INTERVAL == 0:
            self.con.commit()

    def close(self, log):
        self.con.commit()
        self.con.close()
        self.con = None
        log.write("Timing records available in file %s\n" % self.filename)

# An instance of NullTimer can be used in place of Timer, when no data
# collection is wanted.
class NullTimer(object):
    def __init__(self):
        pass
    def start(self, modname, sample):
        pass
    def capture(self):
        pass
    def close(self, log):
        pass



