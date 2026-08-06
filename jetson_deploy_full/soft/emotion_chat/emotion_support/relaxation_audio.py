"""Non-blocking relaxation audio support strategy."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from .models import Emotion, StrategyResult, SupportMethod
from .safety import should_cancel
from .strategy_base import SupportStrategy

LOGGER = logging.getLogger(__name__)


class AudioPlayer:
    """Small non-blocking audio launcher with best-effort stop support."""

    def __init__(self) -> None:
        self._process = None

    def play(self, path: Path) -> None:
        self.stop()
        if sys.platform == "win32":
            self._process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "Start-Process -FilePath $env:AURA_RELAX_AUDIO -PassThru | Select-Object -ExpandProperty Id",
                ],
                env={**os.environ, "AURA_RELAX_AUDIO": str(path)},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        for player in ("ffplay", "aplay", "paplay"):
            try:
                if player == "ffplay":
                    args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
                else:
                    args = [player, str(path)]
                self._process = subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except FileNotFoundError:
                continue

        raise RuntimeError("Không tìm thấy trình phát audio phù hợp.")

    def stop(self) -> None:
        proc = self._process
        self._process = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            LOGGER.debug("Ignoring audio stop error", exc_info=True)


class RelaxationAudioStrategy(SupportStrategy):
    """Let the user choose a configured relaxation audio file."""

    method = SupportMethod.RELAXATION_AUDIO

    def __init__(self, *args, audio_player: Optional[AudioPlayer] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.audio_player = audio_player or AudioPlayer()

    def _audio_map(self) -> Dict[str, str]:
        return dict(self.config.get("relaxation_audio", {}))

    def run(self, emotion: Emotion, intensity: Optional[int]) -> StrategyResult:
        audio_map = self._audio_map()
        if not audio_map:
            self.io.say("Mình chưa có danh sách âm thanh thư giãn để phát.")
            return StrategyResult(completed=False, notes="no_audio_config")

        names = list(audio_map.keys())
        self.io.say("Bạn muốn nghe loại nào: %s?" % ", ".join(names))
        answer = self.io.ask("Chọn âm thanh: ").strip().lower()
        if should_cancel(answer):
            return StrategyResult(completed=False, cancelled=True, notes="user_cancelled")
        if not answer:
            answer = names[0]

        selected = None
        for name in names:
            if answer == name.lower() or answer in name.lower() or name.lower() in answer:
                selected = name
                break
        selected = selected or names[0]
        path = Path(audio_map[selected])
        if not path.is_absolute():
            base_dir = Path(self.config.get("_base_dir", "."))
            path = base_dir / path

        if not path.exists():
            self.io.say("Mình chưa tìm thấy file âm thanh %s, nên sẽ không phát." % selected)
            return StrategyResult(completed=False, notes="missing_audio:%s" % selected)

        try:
            self.audio_player.play(path)
            self.io.say("Mình đã mở âm thanh %s. Khi muốn dừng, bạn nói dừng lại nhé." % selected)
            answer = self.io.ask("Enter để tiếp tục hoặc nói dừng lại: ")
            if should_cancel(answer):
                self.audio_player.stop()
                self.io.say("Mình đã dừng âm thanh.")
                return StrategyResult(completed=False, cancelled=True, notes="audio_stopped")
            return StrategyResult(completed=True, notes="audio:%s" % selected)
        except Exception as exc:
            LOGGER.warning("Relaxation audio failed: %s", exc)
            self.io.say("Mình chưa phát được âm thanh lúc này, nhưng chúng ta vẫn có thể thử cách khác.")
            return StrategyResult(completed=False, notes="audio_error")
