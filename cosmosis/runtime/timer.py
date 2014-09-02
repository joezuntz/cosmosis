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
    def __init__(self, filename, maxrecords, log):
        self.filename = filename
        self.max_records = maxrecords
        self.log = log
        self.con = sqlite3.connect(filename)
        self.con.execute("CREATE TABLE IF NOT EXISTS Samples "
                         "(module TEXT, sample INTEGER, t REAL);")
        self.num_samples = 0
        self.current_record = None

    # Start collecting data for the module 'modname', and for sample
    # 'sample'.
    def start(self, modname, sample):
        self.num_samples += 1
        self.current_record = [ modname, sample, time.time() ]

    # Stop timing for the current sample, and store the record.
    def capture(self):
        self.current_record[2] = time.time() - self.current_record[2]
        self.con.execute("INSERT INTO Samples (module, sample, t) VALUES (?, ?, ?);",
                         self.current_record)
        if self.num_samples % Timer.COMMIT_INTERVAL == 0:
            self.con.commit()
        if self.num_samples >= self.max_records:
            self.nullify()

    # Flush any outstanding records, and close the output file.
    def close(self):
        self.con.commit()
        self.con.close()
        self.con = None
        self.log.write("Timing records available in file %s\n" % self.filename)

    # "Turn off" the timer by turing it into a NullTimer (after flushing
    # any outstanding records).
    def nullify(self):
        self.close()
        self.__class__ = NullTimer
        self.__init__()
        del(self.filename)
        del(self.con)
        del(self.num_samples)
        del(self.max_records)
        del(self.current_record)

# An instance of NullTimer can be used in place of Timer, when no data
# collection is wanted.
class NullTimer(object):
    def __init__(self):
        pass
    def start(self, modname, sample):
        pass
    def capture(self):
        pass
    def close(self):
        pass
    def nullify(self):
        pass



