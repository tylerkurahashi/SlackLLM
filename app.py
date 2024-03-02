import os
import re
import time
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from const import (CHAT_UPDATE_INTERVAL_SEC, OPENAI_API_MODEL,
                   OPENAI_API_TEMPERATURE)

load_dotenv()

class SlackStreamingCallbackHandler(BaseCallbackHandler):
  last_send_time = time.time()
  message = ""
  
  def __init__(self, channel, ts):
    self.channel = channel
    self.ts = ts
    
  def on_llm_new_token(self, token: str, **kwargs):
    self.message += token
    
    now = time.time()
    if now - self.last_send_time > CHAT_UPDATE_INTERVAL_SEC:
      self.last_send_time = now
      app.client.chat_update(
        channel=self.channel, ts=self.ts, text=f"{self.message}..."
      )
      
  def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
    app.client.chat_update(channel=self.channel, ts=self.ts, text=self.message)

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)


@app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*>", "", event["text"])
    
    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]
    
    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)
    llm = ChatOpenAI(
      model=OPENAI_API_MODEL,
      temperature=OPENAI_API_TEMPERATURE,
      streaming=True,
      callbacks=[callback]
    )

    llm.predict(message)


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
