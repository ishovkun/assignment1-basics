import multiprocessing

class WorkerProgress:
    def __init__( self,
        progress_queue: multiprocessing.Queue,
        total: int = 0):
        self.queue = progress_queue
        # self.queue.put(("total", total))
        self.setTotal(total)
    def update(self, count: int):
        self.queue.put(("progress", count))
    def setTotal(self, total: int):
        self.queue.put(("total", total))

class MasterProgress:
    def __init__(self, queue, pbar, num_workers):
        self.queue = queue
        self.pbar = pbar
        self.numWorkers = num_workers
        self.total = 0
    def update(self):
        while not self.queue.empty():
            msg = self.queue.get()
            if msg[0] == "total":
                self.total += msg[1]
                self.pbar.total = self.total
                self.pbar.refresh()
            elif msg[0] == "progress":
                self.pbar.update(msg[1])
