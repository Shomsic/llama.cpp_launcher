import subprocess
import threading
import os
import re
from core.i18n import _

class LlamaBenchManager:
    def __init__(self):
        self.process = None
        self.running = False
        self.lock = threading.Lock()
        self.on_output = None
        self.on_finished = None

    def run(self, exe_path, model_path, prompt=512, predict=128, threads=8, ngl=999, on_output=None, on_finished=None):
        with self.lock:
            if self.running:
                return False, _("already_running")

            self.on_output = on_output
            self.on_finished = on_finished

            cmd = [
                exe_path,
                "-m", model_path,
                "-p", str(prompt),
                "-n", str(predict),
                "-t", str(threads),
                "-ngl", str(ngl),
                "-r", "1"
            ]

            try:
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    startupinfo=startupinfo,
                    errors="replace"
                )
                self.running = True

                threading.Thread(target=self._output_reader, daemon=True).start()
                return True, "OK"
            except Exception as e:
                return False, str(e)

    def _output_reader(self):
        if not self.process:
            return

        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                if self.on_output:
                    self.on_output(line.strip())
        except Exception as e:
            if self.on_output:
                self.on_output(f"Error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        with self.lock:
            self.running = False
            if self.process:
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                    except:
                        pass
                self.process = None
        
        if self.on_finished:
            self.on_finished()

    def stop(self):
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                return True
            return False
