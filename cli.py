"""Interactive CLI to exercise the counseling pipeline turn by turn.

python cli.py --debug True
python cli.py
"""
import argparse
import time

from debug import debug_print, set_debug
from graph import get_graph
from state import new_session
from stt import (
    DEVICE_NAME_PRIORITY,
    find_input_device,
    find_priority_input_device,
    load_model,
    transcribe_stream,
)


def _str2bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def _print_debug(state: dict) -> None:
    print(
        f"  [stage={state['stage']} turn_count={state['turn_count']} "
        f"gate={state['gate']} selected_values={state.get('selected_values')}]"
    )
    for slot, values in state["slots"].items():
        content = " / ".join(values) if values else "X"
        print(f"    - {slot}: {content}")
    print(f"  [pending(target_slot/question_intent)={state.get('pending')}]")
    print(f"  [asked_slots={state.get('asked_slots')}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=_str2bool, default=False)
    parser.add_argument(
        "--device", default=None,
        help="입력 오디오 장치 이름 일부 또는 인덱스 (기본: 헤드셋 > 아이폰 순으로 자동 탐색)",
    )
    args = parser.parse_args()
    set_debug(args.debug)

    if args.device is not None:
        device = int(args.device) if args.device.isdigit() else find_input_device(args.device)
    else:
        device = find_priority_input_device(DEVICE_NAME_PRIORITY)

    stt_model = load_model()
    listen = transcribe_stream(stt_model, device=device, debug=args.debug)

    graph = get_graph()
    name = input("닉네임을 입력해주세요: ").strip()
    state = new_session(name)

    print("[상담을 시작합니다 - 종료하려면 Ctrl+C]\n")
    state = graph.invoke(state)
    print(f"상담사: {state['bot_message']}\n")
    if args.debug:
        _print_debug(state)

    try:
        while state["stage"] != "done":
            print("나: (말씀해주세요...)")
            try:
                user_input, last_loud_time = next(listen)
            except StopIteration:
                break
            print(f"나: {user_input}\n")

            state["user_input"] = user_input
            turn_start = time.perf_counter()
            state = graph.invoke(state)
            elapsed = time.perf_counter() - turn_start
            debug_print(f"[TIMING DEBUG] 응답 생성까지 걸린 시간: {elapsed:.2f}초")

            total_perceived = time.perf_counter() - last_loud_time
            debug_print(f"[TIMING DEBUG] 체감 지연시간(말 멈춤 → 응답 출력, STT+LLM): {total_perceived:.2f}초")

            print(f"\n상담사: {state['bot_message']}\n")
            if args.debug:
                _print_debug(state)
    except KeyboardInterrupt:
        pass
    finally:
        listen.close()

    if state["stage"] == "done":
        print("[상담이 마무리되었습니다. 대화를 종료합니다]")


if __name__ == "__main__":
    main()
