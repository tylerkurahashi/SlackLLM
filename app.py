import os
import re
import time
from typing import Any
from datetime import timedelta

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import MomentoChatMessageHistory
from langchain.schema import HumanMessage, LLMResult, SystemMessage
# from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from const import (
    CHAT_UPDATE_INTERVAL_SEC,
    OPENAI_API_MODEL,
    OPENAI_API_TEMPERATURE,
    MOMENTO_CACHE,
    MOMENTO_TTL,
)

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

    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

    history = MomentoChatMessageHistory.from_client_params(
        id_ts, MOMENTO_CACHE, timedelta(hours=int(MOMENTO_TTL))
    )

    messages = [SystemMessage(conten="You are a good assistant.")]
    messages.extend(history.messages)
    messages.append(HumanMessage(content=message))

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)
    llm = ChatOpenAI(
        model=OPENAI_API_MODEL,
        temperature=OPENAI_API_TEMPERATURE,
        streaming=True,
        callbacks=[callback],
    )

    ai_message = llm(messages)
    history.add_message(ai_message)


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
