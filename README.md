### 실행 방법
---

1. 레포지토리 클론
2. `pume-voice-qwen3-4b-q4_k_m.gguf` 파일을 (용량 문제로 저장소에 미포함) 레포 루트에 직접 추가
3. Ollama에 파인튜닝 모델 등록: `ollama create pume-voice-qwen3-4b -f Modelfile`
4. Python 3.10 이상으로 가상환경 생성 후 활성화 (예: `python3.12 -m venv venv && source venv/bin/activate`)
5. `pip install -r requirements.txt`
6. `.env` 파일 생성 후 아래 값 채움
   ```
   OPENAI_API_KEY=""
   KMP_DUPLICATE_LIB_OK=TRUE
   ```
7. `python -m cli --debug`
