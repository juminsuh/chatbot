### 실행 방법
---

1. 레포지토리 클론
2. Python 3.10 이상으로 가상환경 생성 후 활성화 (예: `python3.12 -m venv venv && source venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `.env` 파일 생성 후 아래 값 채움
   ```
   OPENAI_API_KEY=""
   ```
5. `python -m cli --debug`
6. demo_scenarios.md에 있는 예시들 차례대로 테스트해보기
7. 아직 최종 summary의 퀄리티는 낮음. (map_data.json에 있는 것만 사용하는데, 사용자의 구체적인 상황에 대한 데이터 부족 이슈)
