import os
import logging
import asyncio
import nest_asyncio
import emoji
import re
from telegram import Update, ForceReply
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables from .env
load_dotenv()

# Retrieve Telegram Bot Token from the environment
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the .env file.")

# Import your LLM and translation functions
from LLMmain import get_docs, generate_answer
from trans import sinhalaToEnglish, englishToSinhala

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def is_sinhala(text: str) -> bool:
    """Detect if the text contains Sinhala characters."""
    return any('\u0D80' <= char <= '\u0DFF' for char in text)

def contains_emoji(text: str) -> bool:
    """Check if the text contains emojis."""
    return emoji.emoji_count(text) > 0

def extract_emojis(text: str) -> list:
    """Extract all emojis from the text."""
    return [c for c in text if c in emoji.EMOJI_DATA]

def describe_emoji(emoji_char: str) -> str:
    """Get a description of what the emoji looks like."""
    if emoji_char in emoji.EMOJI_DATA:
        # Get the CLDR short name (description) of the emoji
        return emoji.EMOJI_DATA[emoji_char].get('en', f"emoji {emoji_char}")
    return f"emoji {emoji_char}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}, welcome to the MommyCare Chatbot. You can ask medical questions or share your feelings. For personalized advice, please consult your doctor. Type /help for assistance.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when /help is issued."""
    await update.message.reply_text(
        "Send your question in English or Sinhala. I'll reply with supportive, accurate information along with source references and a disclaimer. You can also use emojis and I'll try to understand what you mean."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming messages from Telegram."""
    user_text = update.message.text.strip()
    logger.info("Received message: %s", user_text)
    
    # Check if message contains emojis
    if contains_emoji(user_text):
        logger.info("Detected emoji in the message")
        emojis_list = extract_emojis(user_text)
        
        # Create descriptions of the emojis
        emoji_descriptions = [describe_emoji(e) for e in emojis_list]
        emoji_desc_text = ", ".join(emoji_descriptions)
        
        # Create a query to interpret the emojis
        if user_text.strip() == "".join(emojis_list):
            # Message contains only emojis
            emoji_query = f"Interpret what the user might be expressing with these emojis: {' '.join(emojis_list)}. The emojis represent: {emoji_desc_text}"
        else:
            # Message contains text and emojis
            emoji_query = f"Interpret the user's message which includes emojis. Message: '{user_text}'. The emojis used are: {' '.join(emojis_list)} which represent: {emoji_desc_text}"
        
        # Process with LLM
        docs = get_docs(emoji_query, top_k=3)
        final_answer = generate_answer(emoji_query, docs)
    
    # Check if message is in Sinhala
    elif is_sinhala(user_text):
        logger.info("Detected Sinhala input. Translating query to English...")
        english_query = sinhalaToEnglish(user_text)
        logger.info("Translated query: %s", english_query)
        docs = get_docs(english_query, top_k=5)
        english_answer = generate_answer(english_query, docs)
        # Translate the answer back to Sinhala
        final_answer = englishToSinhala(english_answer)
    
    # Standard English text processing
    else:
        docs = get_docs(user_text, top_k=5)
        final_answer = generate_answer(user_text, docs)
    
    await update.message.reply_text(final_answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages with an error message."""
    logger.info("Received voice message. Sending error response.")
    await update.message.reply_text(
        "Sorry, I encountered an error processing your voice message. Please try sending your question as text instead."
    )

async def main() -> None:
    """Main function to build and run the bot."""
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add voice message handler
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    logger.info("Bot started. Listening for messages...")
    # Run polling in asynchronous mode.
    await app.run_polling()

if __name__ == '__main__':
    asyncio.run(main())