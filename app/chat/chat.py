import random
from langchain.chat_models import ChatOpenAI
from app.chat.models import ChatArgs
from app.chat.vector_stores import retriever_map
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from app.chat.chains.retrieval import StreamingConversationRetrievalChain
from app.web.api import (
    set_conversation_components,
    get_conversation_components
)
from app.chat.score import random_component_by_score


def select_component(
        component_type, component_map, chat_args
):
    components = get_conversation_components(chat_args.conversation_id)

    previous_component = components[component_type]

    if previous_component:
        # This is NOT the first message of the conversation
        build_component = component_map[previous_component]
        return previous_component, build_component(chat_args)
    else:
        # This is the first message of the conversation
        random_component_name = random_component_by_score(
            component_type, component_map)
        build_component = component_map[random_component_name]
        return random_component_name, build_component(chat_args)


def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """

    # retriever = build_retriever(chat_args)
    retriever_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args
    )

    # llm = build_llm(chat_args)
    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args
    )

    # memory = build_memory(chat_args)
    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args
    )

    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory=memory_name
    )

    print(
        f">> Running chain with:: memory: {memory_name}, llm: {llm_name}, retriever: {retriever_name}"
    )

    condense_question_llm = ChatOpenAI(streaming=False)

    return StreamingConversationRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        condense_question_llm=condense_question_llm,
        retriever=retriever,
        metadata=chat_args.metadata
    )
