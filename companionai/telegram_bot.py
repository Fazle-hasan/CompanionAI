import os
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from agentic_app import build_graph

load_dotenv()

# ------------------- Logging Setup -------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Console handler (INFO and above)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Info file handler — rotates daily at midnight
info_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "info.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
root_logger.addHandler(info_handler)

# Error file handler — rotates daily at midnight
error_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    when="midnight",
    interval=1,
    backupCount=30,
    encoding="utf-8",
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
root_logger.addHandler(error_handler)

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Per-user chat history (in-memory, keyed by Telegram user id)
user_histories: dict[int, list] = {}

HELP_TEXT = (
    "**CompanionAI** — Your Virtual Companion\n\n"
    "Commands:\n"
    "/ask <message> — Ask me anything\n"
    "/newchat — Start a fresh conversation\n"
    "/model <mistral|gemma2|gemma3> — Switch LLM model (default: gemma3)\n"
    "/summarize — Summarize last chat\n"
    "/help — Show this help message\n\n"
    "Or just send me a message directly and I'll respond!"
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Greet the user on /start."""
    await update.message.reply_text(
        "Hey there! I'm CompanionAI, your virtual companion.\n"
        "I'm here to chat, offer emotional support, or just keep you company.\n\n"
        + HELP_TEXT
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show usage instructions on /help."""
    await update.message.reply_text(HELP_TEXT)


async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear per-user chat history on /newchat."""
    user_histories[update.effective_user.id] = []
    await update.message.reply_text("Chat history cleared! Let's start fresh.")


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Switch the LLM model on /model <name>."""
    allowed = ("mistral", "gemma2", "gemma3")
    if context.args and context.args[0].lower() in allowed:
        choice = context.args[0].lower()
        context.user_data["model"] = choice
        await update.message.reply_text(f"Model switched to **{choice}**.")
    else:
        await update.message.reply_text(f"Usage: /model <{'|'.join(allowed)}>")


async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ask <query>."""
    if not context.args:
        await update.message.reply_text("Usage: /ask <your question>")
        return
    query = " ".join(context.args)
    await _process_query(update, context, query)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle plain text messages (no command prefix)."""
    await _process_query(update, context, update.message.text)


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /summarize."""
    user_id = update.effective_user.id
    history = user_histories.get(user_id, [])
    
    await update.message.chat.send_action(ChatAction.TYPING)
    
    if not history:
        await update.message.reply_text("No chat history to summarize.")
        return
        
    history_str = "\n".join([f"Human: {m['human']}\nAI: {m['ai']}" for m in history])
    prompt = f"Please summarize the following conversation concisely:\n\n{history_str}"
    
    model = context.user_data.get("model", "gemma3")
    from ollama import Client
    client = Client(host="http://localhost:11434/")
    try:
        res = client.generate(model=model, prompt=prompt)
        await update.message.reply_text(res['response'])
    except Exception as e:
        logger.error(f"Error summarizing chat: {e}")
        await update.message.reply_text("Failed to summarize chat.")


async def _process_query(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
):
    """Run the agentic RAG pipeline and reply with the answer."""
    user_id = update.effective_user.id
    model = context.user_data.get("model", "gemma2")

    if user_id not in user_histories:
        user_histories[user_id] = []

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        graph = build_graph()
        final_state = graph.invoke(
            {
                "question": query,
                "chat_history": user_histories[user_id],
                "model": model,
                "context": None,
                "answer": None,
                "next": "retrieve",
            }
        )
        user_histories[user_id] = final_state["chat_history"]
        await update.message.reply_text(final_state["answer"])
    except Exception as e:
        logger.error("Error processing query for user %s: %s", user_id, e)
        await update.message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set.\n"
            "1. Talk to @BotFather on Telegram to create a bot and get a token.\n"
            "2. Copy .env.example to .env and paste the token there, or:\n"
            "   export TELEGRAM_BOT_TOKEN='your-token-here'"
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ask", ask_command))
    app.add_handler(CommandHandler("newchat", newchat_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("CompanionAI Telegram bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
