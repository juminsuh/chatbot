
import argparse
import sys
import time
import queue

import numpy as np
import sounddevice as sd
from pywhispercpp.model import ContextParams, Model

# Input device auto-detection priority: first matching hint wins.
DEVICE_NAME_PRIORITY = ["WH-1000XM5", "iphone"]

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.03  # seconds per audio callback block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

SILENCE_THRESHOLD = 0.02   # RMS amplitude below this counts as silence
SHORT_HANG_SEC = 1.5       # first checkpoint: pause length before checking the ending
LONG_HANG_SEC = 4.0        # fallback: end the turn after this much silence no matter what
MIN_SPEECH_SEC = 0.3       # ignore utterances shorter than this (noise blips)
MAX_UTTERANCE_SEC = 15.0   # force-flush very long utterances
PRE_SPEECH_PAD_SEC = 0.3   # keep a little audio from just before speech onset

SPEECH_ONSET_SEC = 0.09    # sustained loudness required to start capture (filters clicks/pops)
MIN_LOUD_FRACTION = 0.15   # fraction of the captured clip that must be above SILENCE_THRESHOLD
                            # -- a real utterance stays loud for a good chunk of its own length;
                            # a noise spike that triggered capture gets diluted by the mostly-quiet
                            # padding/hang tail around it, so this rejects it before it reaches whisper

WHISPER_MODEL_NAME = "small"
LANGUAGE = "ko"
NO_SPEECH_THRESHOLD = 0.6  # forwarded to whisper.cpp's built-in no-speech gate

HALLUCINATION_PHRASES = {"감사합니다"}
INCOMPLETE_ENDING_SUFFIXES = (
    "데", "서", "니까", "으니까",  # "데"/"서" cover -는데/-은데/-인데/-던데 and -아서/-어서/-여서/-워서 etc.
    "고", "지만", "거나", "든지", "면서",
    "은", "는", "이", "가", "을", "를", "에", "에서", "로", "와", "과", "도", "만",
)
INCOMPLETE_FILLER_WORDS = {"음", "어", "그", "저", "그니까", "그러니까", "그래서", "이제", "저는", "그게", "근데", "그리고"}


def looks_incomplete(text: str) -> bool:
    normalized = text.strip().rstrip(".!?~ ")
    if not normalized:
        return True
    last_word = normalized.split()[-1]
    if last_word in INCOMPLETE_FILLER_WORDS:
        return True
    return normalized.endswith(INCOMPLETE_ENDING_SUFFIXES)


def _rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block))))


def is_hallucination_phrase(text: str) -> bool:
    normalized = text.strip().strip(".!?~ ")
    return normalized in HALLUCINATION_PHRASES


def find_input_device(name_hint: str) -> int | None:
    """Returns the index of the first input-capable device whose name
    contains name_hint (case-insensitive), or None if none matches."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name_hint.lower() in dev["name"].lower():
            return idx
    return None


def find_priority_input_device(name_hints: list[str]) -> int | None:
    """Tries each hint in order and returns the index of the first match."""
    for hint in name_hints:
        idx = find_input_device(hint)
        if idx is not None:
            return idx
    return None


def microphone_utterances(model: Model, sample_rate: int = SAMPLE_RATE, device: int | None = None):
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stt] audio status: {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    pre_buffer: list[np.ndarray] = []
    pre_buffer_max = int(PRE_SPEECH_PAD_SEC / BLOCK_DURATION)

    speech_buffer: list[np.ndarray] = []
    speaking = False
    silence_run = 0.0
    onset_run = 0.0
    speech_len = 0.0
    last_loud_time = 0.0  # wall-clock time of the most recent loud block in this utterance
    checkpoint_done = False  # whether the short-pause ending check already ran for this silence run

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        device=device,
        callback=callback,
    ):
        device_name = sd.query_devices(device)["name"] if device is not None else sd.query_devices(kind="input")["name"]
        print(f"[stt] '{device_name}' 마이크 대기 중... (Ctrl+C로 종료)")
        while True:
            block = audio_q.get()
            level = _rms(block)

            if level >= SILENCE_THRESHOLD:
                silence_run = 0.0
                checkpoint_done = False
                last_loud_time = time.perf_counter()
                if speaking:
                    speech_buffer.append(block)
                    speech_len += len(block) / sample_rate
                else:
                    onset_run += BLOCK_DURATION
                    pre_buffer.append(block)
                    if len(pre_buffer) > pre_buffer_max:
                        pre_buffer.pop(0)
                    if onset_run >= SPEECH_ONSET_SEC:
                        speaking = True
                        speech_buffer = list(pre_buffer)
                        speech_len = sum(len(b) for b in speech_buffer) / sample_rate
            else:
                onset_run = 0.0
                pre_buffer.append(block)
                if len(pre_buffer) > pre_buffer_max:
                    pre_buffer.pop(0)
                if speaking:
                    speech_buffer.append(block)
                    speech_len += len(block) / sample_rate
                    silence_run += BLOCK_DURATION

            if not speaking:
                continue

            should_finalize = speech_len >= MAX_UTTERANCE_SEC or silence_run >= LONG_HANG_SEC

            if not should_finalize and not checkpoint_done and silence_run >= SHORT_HANG_SEC:
                checkpoint_done = True
                checkpoint_segments = model.transcribe(np.concatenate(speech_buffer), language=LANGUAGE)
                checkpoint_text = "".join(s.text for s in checkpoint_segments).strip()
                if not looks_incomplete(checkpoint_text):
                    should_finalize = True

            if should_finalize:
                speaking = False
                silence_run = 0.0
                onset_run = 0.0
                checkpoint_done = False
                pre_buffer = []
                loud_fraction = sum(1 for b in speech_buffer if _rms(b) >= SILENCE_THRESHOLD) / len(speech_buffer)
                should_yield = speech_len >= MIN_SPEECH_SEC and loud_fraction >= MIN_LOUD_FRACTION
                utterance_audio = np.concatenate(speech_buffer) if should_yield else None
                speech_buffer = []
                speech_len = 0.0
                if should_yield:
                    yield utterance_audio, last_loud_time
                    # the caller (e.g. an LLM turn) may have taken a while to ask
                    # for the next utterance -- anything captured in the meantime
                    # is a leftover from before this turn, not an answer to it
                    while not audio_q.empty():
                        try:
                            audio_q.get_nowait()
                        except queue.Empty:
                            break


def load_model() -> Model:
    model = Model(
        WHISPER_MODEL_NAME,
        context_params=ContextParams(use_gpu=True),
        language=LANGUAGE,
        no_speech_thold=NO_SPEECH_THRESHOLD,
        print_progress=False,
    )
    print(f"[stt] '{WHISPER_MODEL_NAME}' loaded (see ggml_metal_init log above for GPU backend status)")
    return model


def transcribe_stream(model: Model, device: int | None = None, debug: bool = False):
    for audio, last_loud_time in microphone_utterances(model, device=device):
        start = time.perf_counter()
        segments = model.transcribe(audio, language=LANGUAGE)
        text = "".join(segment.text for segment in segments).strip()
        stt_elapsed = time.perf_counter() - start
        if debug:
            # perceived latency: from the user's last loud block to text-ready,
            # i.e. the endpoint wait (SHORT_HANG_SEC or LONG_HANG_SEC, plus any
            # checkpoint transcribe calls) plus this final STT call
            perceived_latency = time.perf_counter() - last_loud_time
            silence_wait = perceived_latency - stt_elapsed
            print(
                f"[stt][debug] STT 체감 지연시간: {perceived_latency:.2f}s "
                f"(침묵 대기 {silence_wait:.2f}s + STT 처리 {stt_elapsed:.2f}s)",
                file=sys.stderr,
            )
        if text and is_hallucination_phrase(text):
            print(f"[stt] discarded suspected hallucination: {text!r}", file=sys.stderr)
            continue
        if text:
            yield text, last_loud_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default=None,
        help="입력 오디오 장치 이름 일부 또는 인덱스 (기본: 헤드셋 > 아이폰 순으로 자동 탐색)",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--debug", action="store_true", help="음성 입력 후 STT 처리 시간을 출력")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.device is not None:
        device = int(args.device) if args.device.isdigit() else find_input_device(args.device)
    else:
        device = find_priority_input_device(DEVICE_NAME_PRIORITY)
        if device is None:
            print(
                f"[stt] {DEVICE_NAME_PRIORITY} 중 일치하는 마이크를 찾지 못해 기본 입력 장치를 사용합니다. "
                "--list-devices로 사용 가능한 장치를 확인할 수 있습니다.",
                file=sys.stderr,
            )

    model = load_model()
    for text, _ in transcribe_stream(model, device=device, debug=args.debug):
        print(f"나: {text}")


if __name__ == "__main__":
    main()
