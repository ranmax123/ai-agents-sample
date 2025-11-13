import asyncio

from temporalio.client import Client
from temporalio.contrib.openai_agents import OpenAIAgentsPlugin

from  .workflows.research_bot_workflow import ResearchWorkflow


async def main():
    # Create client connected to server at the given address
    client = await Client.connect(
        "localhost:7233",
        plugins=[
            OpenAIAgentsPlugin(),
        ],
    )

    # Execute a workflow
    result = await client.execute_workflow(
        ResearchWorkflow.run,
        "Goa in January - weather, homestay options, vacation spots and water sports",
        id="research-workflow",
        task_queue="openai-agents-task-queue",
    )

    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
