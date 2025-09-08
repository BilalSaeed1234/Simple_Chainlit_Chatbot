import os
import asyncio
from typing import cast, Dict, List
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# === Chat Start ===
@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=3,
        timeout=30.0
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("conversation", {
        "user_messages": [],
        "assistant_messages": [],
        "full_history": [],
        "user_info": {}
    })

    cl.user_session.set("config", config)

    agent = Agent(
        name="Bilal Saeed AI Assistant",
        instructions="""You are a helpful AI assistant. Follow these rules:
1. Remember personal details like names when shared
2. For history questions:
   - "what was my last message" → "Your last message was: [message]"
   - "what was your last message" → "My last message was: [message]"
3. For "what is my name?" → Respond with stored name
4. Keep responses natural and helpful"""
        ,
        model=model
    )

    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome to the Bilal Saeed AI Assistant! How can I help you today?").send()

# === Chat Message Handler ===
@cl.on_message
async def main(message: cl.Message):
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    conv: Dict = cl.user_session.get("conversation")

    user_input = message.content.strip()
    conv["user_messages"].append(user_input)
    conv["full_history"].append({"role": "user", "content": user_input})

    try:
        lowered = user_input.lower()

        # Store name
        if "my name is" in lowered:
            name = user_input.split("my name is", 1)[1].strip()
            conv["user_info"]["name"] = name
            response = f"Nice to meet you, {name}! I'll remember that."

        # Retrieve name
        elif "what is my name" in lowered:
            name = conv["user_info"].get("name", "")
            response = f"Your name is {name}" if name else "You haven't told me your name yet"

        # User's last message
        elif "what was my last message" in lowered:
            response = (
                f"Your last message was: {conv['user_messages'][-2]}"
                if len(conv["user_messages"]) > 1 else
                "No previous messages"
            )

        # Assistant's last message
        elif "what was your last message" in lowered:
            response = (
                f"My last message was: {conv['assistant_messages'][-1]}"
                if conv["assistant_messages"] else
                "I haven't responded yet"
            )

        else:
            # Simulate streaming AI response
            msg = cl.Message(content="")
            await msg.send()

            result = await Runner.run(
                agent,
                input=user_input,
                run_config=config
            )

            response = result.final_output

            for i in range(0, len(response), 3):
                await msg.stream_token(response[i:i+3])
                await asyncio.sleep(0.05)

            await msg.update()

        # Save assistant response
        conv["assistant_messages"].append(response)
        conv["full_history"].append({"role": "assistant", "content": response})

        if "what" in lowered or "my name is" in lowered:
            await cl.Message(content=response).send()

    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        await cl.Message(content=error_msg).send()

        # Rollback last user message from history
        conv["user_messages"].pop()
        conv["full_history"].pop
