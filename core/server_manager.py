import subprocess
import threading
import socket
import os
import shlex
import re
import time
from core.i18n import _

# Pre-compiled ANSI escape code pattern for stripping color codes from server output
_ANSI_ESCAPE_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class LlamaServerManager:
    def __init__(self):
        self.process = None
        self.running = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.on_log = None
        self.on_stop = None
        self.fallback_args = None
        self.has_fallen_back = False

    def start(self, cmd_args, cwd=None, on_log=None, on_stop=None, fallback_args=None):
        with self.lock:
            if self.running:
                return False, _("already_running")

            self.on_log = on_log
            self.on_stop = on_stop
            self.stop_event.clear()
            self.fallback_args = fallback_args
            self.has_fallen_back = False

            try:
                # В Windows прячем окно консоли
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                self.process = subprocess.Popen(
                    cmd_args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=cwd,
                    startupinfo=startupinfo,
                    errors="replace"
                )
                self.running = True

                # Запускаем поток чтения логов
                threading.Thread(target=self._log_reader, daemon=True).start()
                return True, "OK"
            except Exception as e:
                return False, str(e)

    def _log_reader(self):
        if not self.process:
            return

        try:
            for line in iter(self.process.stdout.readline, ''):
                if self.stop_event.is_set():
                    break
                if not line:
                    break

                # Чистим ANSI-коды цветов
                clean_line = _ANSI_ESCAPE_RE.sub('', line).rstrip()

                if clean_line:
                    # Detect failure for fallback
                    if not self.has_fallen_back and self.fallback_args:
                        if any(err in clean_line.lower() for err in ["unsupported architecture", "failed to load model", "error loading model"]):
                            if self.on_log:
                                self.on_log(_("detected_failure_falling_back").format(clean_line))
                            
                            # Trigger fallback
                            self.stop()
                            # Start with fallback args
                            # Note: we need to be careful about the lock here. stop() acquires it.
                            # We use a separate thread to restart to avoid deadlocks in _log_reader
                            threading.Thread(target=self._do_fallback, daemon=True).start()
                            break

                    if self.on_log:
                        self.on_log(clean_line)

        except Exception as e:
            if self.on_log:
                self.on_log(_("log_read_error").format(e))
        finally:
            self._cleanup_process()

    def _do_fallback(self):
        self.has_fallen_back = True
        # Small delay to ensure stop() finished
        time.sleep(1)
        if self.fallback_args:
            # Re-start with fallback args, using the same callbacks
            self.start(self.fallback_args, on_log=self.on_log, on_stop=self.on_stop)

    def _cleanup_process(self):
        """Гарантированная очистка процесса."""
        with self.lock:
            self.running = False
            if not self.process:
                return

            # Закрываем stdout
            try:
                self.process.stdout.close()
            except Exception:
                pass

            # Если процесс ещё жив — убиваем
            if self.process.poll() is None:
                try:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=1.0)
                except Exception:
                    pass

            self.process = None

        if self.on_stop:
            self.on_stop()

    def stop(self):
        with self.lock:
            if not self.running or not self.process:
                return False

            self.stop_event.set()
            self.running = False
            try:
                self.process.terminate()
                # Даем время на мягкое закрытие
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            except Exception as e:
                if self.on_log:
                    self.on_log(_("stop_error").format(e))
            
            return True

    @staticmethod
    def check_port_available(host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, int(port)))
                return True
            except OSError:
                return False
