from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

NOT_FOUND_ANSWER = "В базе знаний нет информации для ответа на этот вопрос."
SELECTED_CONTEXT_MESSAGE_LIMIT = 8

ANSWER_SYSTEM_PROMPT = (
    "Ты консультант по правилам международной стажировки. "
    "Отвечай только на основе контекста. "
    "Учитывай историю диалога только для понимания текущего вопроса. "
    f"Если ответа нет в контексте, скажи: {NOT_FOUND_ANSWER}"
)

CLARIFICATION_SYSTEM_PROMPT = (
    "Ты помощник, который задает короткий уточняющий вопрос. "
    "Не отвечай на исходный вопрос. "
    "Попроси пользователя выбрать один вариант из списка label. "
    "Используй только переданные label. "
    "Пиши просто и лаконично."
)

ANSWER_HUMAN_PROMPT_TEMPLATE = "Контекст:\n{context}\n\nВопрос:\n{question}"

CLARIFICATION_HUMAN_PROMPT_TEMPLATE = (
    "Группа уточнения: {group}\n"
    "Исходный вопрос пользователя: {question}\n"
    "Label, которые нужно уточнить: {labels}"
)

LLM_ERROR_MESSAGE = "Ошибка вызова LLM"
CLARIFICATION_LLM_ERROR_MESSAGE = "Ошибка вызова LLM для уточнения"
