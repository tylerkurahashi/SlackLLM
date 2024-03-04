import os
import json
import logging
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
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

from const import (
    CHAT_UPDATE_INTERVAL_SEC,
    OPENAI_API_MODEL,
    OPENAI_API_TEMPERATURE,
    MOMENTO_CACHE,
    MOMENTO_TTL,
)

load_dotenv()

# logging
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s  [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs):
        self.message += token

        now = time.time()
        if now - self.last_send_time > CHAT_UPDATE_INTERVAL_SEC:
            self.last_send_time = now
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f"{self.message}..."
            )

            self.last_send_time = now
            self.update_count += 1

            if self.update_count / 10 > self.interval:
                self.interval *= 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = (
            "ChatGPT can make mistakes. Consider checking important information."
        )
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,  # type: ignore
        )


app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)


def handler(event, context):
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return 200

    slack_handler = SlackRequestHandler(app)
    return slack_handler.handle(event, context)


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

    messages = [SystemMessage(content="You are a good assistant.")]
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


def just_ack(ack):
    ack()


app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
