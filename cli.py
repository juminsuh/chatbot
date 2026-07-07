"""Interactive CLI to exercise the counseling pipeline turn by turn.

Run from repo root: `python -m counsel.cli [--debug]`
"""
import argparse

from graph import get_graph
from state import new_session


def _print_debug(state: dict) -> None:
    filled = [slot for slot, values in state["slots"].items() if values]
    print(
        f"  [stage={state['stage']} turn_count={state['turn_count']} off_topic={state['off_topic']} "
        f"filled={filled} pattern_final={state['pattern_final']} gate={state['gate']}]"
    )
    print(
        f"  [supervisor_action={state.get('supervisor_action')} "
        f"pending(qwen에 전달된 target_slot/question_intent)={state.get('pending')}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    graph = get_graph()
    state = new_session()

    print("[상담을 시작합니다 - 종료하려면 'quit' 입력]\n")
    state = graph.invoke(state)
    print(f"상담사: {state['bot_message']}\n")
    if args.debug:
        _print_debug(state)

    while state["stage"] != "done":
        try:
            user_input = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit"):
            break

        state["user_input"] = user_input
        state = graph.invoke(state)
        print(f"\n상담사: {state['bot_message']}\n")
        if args.debug:
            _print_debug(state)

    if state["stage"] == "done":
        print("[상담이 마무리되었습니다. 대화를 종료합니다]")


if __name__ == "__main__":
    main()
