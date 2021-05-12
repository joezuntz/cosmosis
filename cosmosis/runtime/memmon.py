import time
import psutil
import threading
import datetime


class MemoryMonitor:
    """
    A monitor which reports on memory usage by this process throughout the lifetime of
    a process.

    The monitor is designed to be run in a thread, which is done automatically in the
    start_in_thread method, and will then continue until either the main thread ends
    or the stop method is called from another thread.

    To print out different process information you could use subclass and override the
    log method.
    """

    def __init__(self, interval=30):
        """Create a memory monitor.

        Parameters
        ----------
        interval: float or int
            The interval in seconds between each report.
            Default is 30 seconds
        """
        self.should_continue = True
        self.interval = interval
        self.process = psutil.Process()

    @classmethod
    def start_in_thread(cls, *args, **kwargs):
        """Create a new thread and run the memory monitor in it

        For parameters, see the init method; all arguments sent to this method are
        passed directly to it.

        Returns
        -------
        monitor: MemoryMonitor
            The monitor, already running in its own thread
        """
        monitor = cls(*args, **kwargs)
        thread = threading.Thread(target=monitor._run)
        thread.start()
        return monitor

    def stop(self):
        """Stop the monitor.

        The monitor will complete its current sleep interval and then end.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.should_continue = False

    @staticmethod
    def log(p):
        """Print memory usage information to screen

        By default this method prints the p

        Parameters
        ----------
        p: Process
            A psutil process
        """
        mem = p.memory_info()
        # report time since start of process
        dt = datetime.timedelta(seconds=time.time() - p.create_time())

        # Various memory
        rss = mem.rss / 1e6
        vms = mem.vms / 1e6
        avail = psutil.virtual_memory().available / 1e6

        # For now I don't use the python logging mechanism, but
        # at some point should probably switch to that.
        print(
            f"MemoryMonitor Time: {dt}   Physical mem: {rss:.1f} MB   "
            f"Virtual mem: {vms:.1f} MB   "
            f"Available mem: {avail:.1f} MB"
        )

    def _run(self):
        # there are two ways to stop the monitor - it is automatically
        # ended if the main thread completes.  And it can be stopped
        # manually using the stop method.  Check for both these.
        while threading.main_thread().is_alive():
            if not self.should_continue:
                break
            self.log(self.process)
            time.sleep(self.interval)
