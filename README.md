### Runnging Guidance
---

1. Clone the repository. 
2. Create the virtual enviroment and install what you needed. 
   ```
   conda create -n pme python=3.10 -y
   conda activate pme
   pip install -r requirements.txt
   ```
3. Create `.env` and fill the key. (Please note you should add `.env` into `.gitignore`) 
   ```
   OPENAI_API_KEY=""
   ```
4. `python download_stt.py`: faster-whister-small (244M) will be downloaded. ⚠️ Before running this, you are sure to have Mac Apple Sillicon. ⚠️
5. Connect your headphone with your computer. You can identify the devices list via running `python stt.py --list-devices`
6. Now, you're ready to run the main loop! Please run `python cli.py` (If you want a debug mode which prints out session state or time, flag it with `--debug True`)
7. You can test a live-demo! Feel free to talk your hard or bad things to our chatbot 😊
