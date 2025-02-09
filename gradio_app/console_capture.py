import sys
import io
import threading
import queue

class ConsoleCapture:
    def __init__(self):
        self.output = []
        self.queue = queue.Queue()
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._stop_flag = False
        self._thread = None

    def _write(self, text):
        self.queue.put(text)
        self._stdout.write(text)

    def _start_processing(self):
        while not self._stop_flag:
            try:
                text = self.queue.get(timeout=0.1)
                self.output.append(text)
            except queue.Empty:
                continue

    def start(self):
        """Start capturing console output"""
        self._stop_flag = False
        sys.stdout = type('', (), {'write': self._write, 'flush': lambda: None})()
        sys.stderr = type('', (), {'write': self._write, 'flush': lambda: None})()
        self._thread = threading.Thread(target=self._start_processing)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop capturing and restore original stdout/stderr"""
        self._stop_flag = True
        if self._thread:
            self._thread.join()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

    def get_output(self):
        """Get captured output and clear the buffer"""
        output = self.output.copy()
        self.output.clear()
        return output