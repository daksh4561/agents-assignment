import os
import logging
from dotenv import load_dotenv
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import silero

# FORCE load .env from current file directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

print("ENV CHECK:")
print("LIVEKIT_URL =", os.getenv("LIVEKIT_URL"))
print("LIVEKIT_API_KEY =", os.getenv("LIVEKIT_API_KEY"))
print("LIVEKIT_API_SECRET =", os.getenv("LIVEKIT_API_SECRET"))

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    metrics,
    room_io,
)

from livekit.agents.llm import function_tool
import asyncio

logger = logging.getLogger("intelligent-interruption-agent")
logger.setLevel(logging.INFO)

# ============================================================================
# CONFIGURABLE IGNORE LIST - Modify these as needed
# ============================================================================

# Passive acknowledgment words - IGNORED when agent is speaking
IGNORE_WORDS = {
    "yeah", "ok", "okay", "hmm", "right", "uh-huh", "aha", 
    "yep", "yup", "mhm", "mm-hmm", "sure", "alright", "gotcha"
}

# Active interruption commands - ALWAYS stop the agent
INTERRUPT_WORDS = {
    "stop", "wait", "no", "cancel", "hold", "pause", "hang on", "hold on"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation"""
    normalized = text.lower().strip()
    for char in [".", ",", "!", "?", ":", ";", "-", "'", '"']:
        normalized = normalized.replace(char, "")
    return normalized


def is_only_passive_acknowledgment(text: str) -> bool:
    """Check if input consists ONLY of passive acknowledgment words"""
    normalized = normalize_text(text)
    words = normalized.split()
    
    if len(words) == 0:
        return False
    
    return all(word in IGNORE_WORDS for word in words)


def contains_interrupt_command(text: str) -> bool:
    """Check if text contains any interrupt command words"""
    normalized = normalize_text(text)
    words = normalized.split()
    return any(word in INTERRUPT_WORDS for word in words)


def should_interrupt(text: str, agent_is_speaking: bool) -> bool:
    """
    Core decision logic: Determine if the agent should be interrupted.
    
    Logic Matrix:
    - If agent is NOT speaking: ALWAYS respond
    - If agent IS speaking:
      - Contains interrupt command: INTERRUPT
      - Only passive words: IGNORE
      - Anything else: INTERRUPT
    """
    if not agent_is_speaking:
        logger.info(f"✓ RESPOND: Agent is silent - will process '{text}'")
        return True
    
    if contains_interrupt_command(text):
        logger.info(f"🛑 INTERRUPT: Command detected in '{text}'")
        return True
    
    if is_only_passive_acknowledgment(text):
        logger.info(f"✓ IGNORE: Passive acknowledgment '{text}' - Agent continues")
        return False
    
    logger.info(f"🛑 INTERRUPT: Active input detected '{text}'")
    return True


# ============================================================================
# AGENT CLASS WITH BUILT-IN FILTERING
# ============================================================================

class IntelligentAgent(Agent):
    """
    AI Agent with intelligent interruption handling.
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact with users via voice. "
                "Keep responses concise and natural. "
                "Do not use emojis, asterisks, markdown, or special characters in your speech. "
                "You are friendly, helpful, and speak naturally in English. "
                "When explaining complex topics, break them into digestible parts. "
                "If the user gives short acknowledgments like 'yeah', 'okay', or 'right', "
                "understand they are listening and continue your explanation naturally without stopping."
            )
        )

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str = "", longitude: str = ""
    ):
        """Get weather information for a location"""
        logger.info(f"Weather lookup requested for: {location}")
        return f"The weather in {location} is sunny with a temperature of 70 degrees Fahrenheit."


# ============================================================================
# SERVER SETUP
# ============================================================================

def prewarm(proc: JobProcess):
    """Preload models into process memory for faster session startup"""
    logger.info("Prewarming: Loading VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarming complete")


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Main entry point for each RTC session"""
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting new session in room: {ctx.room.name}")

    # ========================================================================
    # SESSION CONFIGURATION
    # ========================================================================
    
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()
    
    # Track agent speaking state
    agent_speaking_state = {"is_speaking": False, "last_user_input": ""}

    # ========================================================================
    # EVENT HANDLERS - THE KEY TO SOLVING THIS PROBLEM
    # ========================================================================
    
    @session.on("user_started_speaking")
    def on_user_started_speaking(transcript: str):
        """
        Critical handler: This fires when user speech is detected.
        We need to decide IMMEDIATELY if we should stop the agent.
        """
        agent_speaking_state["last_user_input"] = transcript
        is_agent_speaking = (session.agent_state == "speaking")
        
        logger.info(f"🎤 User input detected: '{transcript}' | Agent state: {session.agent_state}")
        
        # Quick check if this is passive acknowledgment
        if is_agent_speaking and is_only_passive_acknowledgment(transcript):
            logger.info(f"✓ IGNORE: Passive acknowledgment detected - NOT stopping agent")
            # DO NOT call session.stop_speaking() here
            # Let the agent continue naturally
            return
        
        # Check for interrupt commands
        if is_agent_speaking and contains_interrupt_command(transcript):
            logger.info(f"🛑 INTERRUPT: Command word detected - STOPPING agent")
            session.stop_speaking()
            return
        
        # Check for real interruptions (not just passive acknowledgments)
        if is_agent_speaking and not is_only_passive_acknowledgment(transcript):
            logger.info(f"🛑 INTERRUPT: Active input detected - STOPPING agent")
            session.stop_speaking()
            return
        
        # If agent is not speaking, process normally
        if not is_agent_speaking:
            logger.info(f"✓ RESPOND: Agent is silent - will process input")

    @session.on("agent_started_speaking")
    def on_agent_started_speaking():
        """Track when agent starts speaking"""
        agent_speaking_state["is_speaking"] = True
        logger.info("🗣️ Agent STARTED speaking")
    
    @session.on("agent_stopped_speaking")
    def on_agent_stopped_speaking():
        """Track when agent stops speaking"""
        agent_speaking_state["is_speaking"] = False
        logger.info("🔇 Agent STOPPED speaking")
        
        # Check if we should respond to the last user input
        last_input = agent_speaking_state["last_user_input"]
        if last_input and not is_only_passive_acknowledgment(last_input):
            logger.info(f"→ Agent stopped, will process: '{last_input}'")

    @session.on("user_speech_committed")
    def on_user_speech_committed(transcript: str):
        """
        This fires when the full transcript is committed.
        Good for logging final results.
        """
        logger.info(f"✅ Final transcript committed: '{transcript}'")

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent):
        """Collect and log usage metrics"""
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        """Log final usage statistics when session ends"""
        summary = usage_collector.get_summary()
        logger.info(f"📊 Session ended. Usage summary: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ========================================================================
    # START SESSION
    # ========================================================================
    
    logger.info("🚀 Starting intelligent agent session...")
    logger.info("=" * 60)
    logger.info("BEHAVIOR:")
    logger.info("- Agent speaking + user says 'yeah/ok/hmm' → AGENT CONTINUES")
    logger.info("- Agent speaking + user says 'stop/wait' → AGENT STOPS")
    logger.info("- Agent silent + user says anything → AGENT RESPONDS")
    logger.info("=" * 60)
    
    # Create agent instance
    agent = IntelligentAgent()
    
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions()
        ),
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LiveKit Intelligent Interruption Agent")
    logger.info("=" * 60)
    logger.info(f"Ignore words: {IGNORE_WORDS}")
    logger.info(f"Interrupt words: {INTERRUPT_WORDS}")
    logger.info("=" * 60)
    
    cli.run_app(server)