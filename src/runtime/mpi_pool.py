import abc
from mpi4py import MPI

class ClosePoolMessage(object):
	def __repr__(self):
		return "<Close pool message>"
	pass

class MPIPool(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, comm, debug=False):
		self.comm = comm
		self.rank = comm.Get_rank()
		self.size = comm.Get_size() - 1
		self.debug = debug
		if self.size==0:
			raise ValueError("Tried to create an MPI pool, but there was only one MPI process available.  Need at least two.")
	def is_master(self):
		return self.rank==0

	def wait(self):
		if self.is_master():
			raise RuntimeError("Master node told to await jobs")
		status = MPI.Status()
		while True:
			if self.debug: print "Worker waiting for task"
			task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
			if self.debug: print "Worker got task %r with tag %d" % (task, status.tag)
			if isinstance(task,ClosePoolMessage):
				if self.debug: print "Worker %d told to quit" % self.rank
				break
			result = self.process_task(task)
			if self.debug: print "Worker sending answer %r with tag %d" % (result, status.tag)
			self.comm.isend(result, dest=0, tag=status.tag) #Return result async

	@abc.abstractmethod
	def process_task(self, task):
		""" Subclass must override this method with the task to be performed """
		return 2*task

	def map(self, dummy_function, tasks):
		ntask = len(tasks)
		#Should be called on the master only
		if not self.is_master():
			self.wait()
			return
		#Send all the tasks off.  Do not wait for them to be received, just continue.
		for i,task in enumerate(tasks):
			worker = i%self.size + 1
			if self.debug: print "Sent task %r to worker %d with tag %d" % (task, worker, i)
			self.comm.isend(task, dest=worker, tag=i)
		results = []
		for i in xrange(ntask):
			worker = i%self.size+1
			if self.debug: print "Master waiting for answer from worker %d with tag %d" % (worker, i)
			result = self.comm.recv(source=worker, tag=i)
			results.append(result)
		return results
	def close(self):
		if self.is_master():
			for i in xrange(self.size):
				self.comm.isend(ClosePoolMessage(), dest=i+1)

