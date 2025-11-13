from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio.client import Client
from temporalio.contrib.openai_agents import ModelActivityParameters, OpenAIAgentsPlugin
from temporalio.worker import Worker

from .workflows.research_bot_workflow import ResearchWorkflow
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from agents import set_tracing_export_api_key
tracing_api_key = os.environ["OPENAI_API_KEY"]
set_tracing_export_api_key(tracing_api_key)

async def main():
    # Create client connected to server at the given address
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(
                model_params=ModelActivityParameters(
                    start_to_close_timeout=timedelta(seconds=120)
                )
            ),
        ],
    )

    worker = Worker(
        client,
        task_queue="openai-agents-task-queue",
        workflows=[
            ResearchWorkflow,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
