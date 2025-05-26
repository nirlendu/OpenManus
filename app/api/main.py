import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.manus import Manus
from app.schema import Message

app = FastAPI()


class PromptRequest(BaseModel):
    prompt: str


@app.post("/api/v1/prompt")
async def stream_prompt(request: PromptRequest):
    """Stream the agent's response to a prompt with human-readable steps"""
    try:
        # Create a new Manus instance
        agent = await Manus.create()

        async def generate():
            try:
                # Add user message to memory
                user_msg = Message.user_message(request.prompt)
                agent.memory.add_message(user_msg)

                # Process the prompt iteratively
                while agent.state not in ["FINISHED", "ERROR"]:
                    # Think step
                    if not await agent.think():
                        break

                    # Act step
                    try:
                        result = await agent.act()
                        if result:
                            # Format the result in a human-readable way
                            yield f"data: {json.dumps({'content': f'Action taken: {result}'})}\n\n"

                            # Check for termination
                            if (
                                isinstance(result, str)
                                and result.strip() == '{"status":"success"}'
                            ):
                                yield f"data: {json.dumps({'content': 'Task completed successfully!'})}\n\n"
                                break
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        break

                    # No need for explicit observe step as it's handled in think()

                # Yield the last assistant message if available
                if agent.memory.messages:
                    for msg in reversed(agent.memory.messages):
                        if msg.role == "assistant" and msg.content:
                            yield f"data: {json.dumps({'content': f'Final thoughts: {msg.content}'})}\n\n"
                            break

            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                await agent.cleanup()

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
