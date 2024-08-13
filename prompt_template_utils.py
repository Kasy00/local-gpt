"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.
system_prompt = """Plik powinien być związany z kostką OLAP i służy do opisu struktury miar i wymiarów. Przeanalizuj plik i odpowiedz na to pytanie w formie obiektu JSON, zawierającego 4 listy przypisane do następujących pól (values, rows, columns and filters). Pamiętaj że ma być to prawidłowy obiekt typu JSON bez opisu i znaków escape. Tak abym łatwo mógł sparsować tego JSONa. Do list mogą trafiać jedynie wartości które są przypisane do kluczy uniqueName. Wpisuj je w cudzysłowiu, pojedynczo do listy i przedzielaj przecinkami. Masz nie dodawać do listy nic innego prócz wartości przypisanych do uniqueName. To znaczy że każda z 4 list ma zawierać tylko array z wypisanymi po przecinku wartościami uniqueName które pasują do zapytania użytkownika. Elementy muszą pochodzić z załączonego pliku. Jeśli w pliku nie ma elementów, które by odpowiadały wymaganiom zapytania, nie dodawaj nic od siebie, możesz zwrócić puste listy. Poza tym obiektem z 4 listami nie generuj ŻADNEGO dodatkowego tekstu, to bardzo istotne. Całość odpowiedzi ma składać się tylko z tego obiektu w formacie JSON. Pamiętaj że dana miara lub wymiar może być użyta tylko raz, a więc nie może się powtarzać w innej liście. Nie opisuj swojej odpowiedzi i nie dodawaj żadnych komentarzy dotyczących zawartości list. Nie wpisuj żadnych wartości zaczynających się na [w_, [WYM_, DIMENSIONS, MEASURES, __MEASURES__, nigdy nie wpisuj całego obiektu typu `unigueName: []`. Wpisuj tylko te które są przypisane do uniqueName po dwukropku i znajdują się w bracketach []. Twoja odpowiedź powinna być wyłącznie obiektem JSON o następującej strukturze z odpowiednimi wartościami pasującymi do utworzenia raportu zgodnego z pytaniem, przykładowa odpowiedź: values: [], rows: [], columns: [], filters: []. W zależności od pytania użytkownika wypełnij listy pasującymi wartościami z pliku tak aby dało się utworzyć dany raport i tabele. Zapytanie jest następujące:"""


def get_prompt_template(system_prompt=system_prompt, promptTemplate_type="llama3", history=False):
    if promptTemplate_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "llama3":

        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"
        B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|> ", "<|eot_id|>"
        ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
        SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = SYSTEM_PROMPT + B_INST + instruction + ASSISTANT_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif promptTemplate_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt
                + """
            
            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    else:
        # change this based on the model you have selected.
        if history:
            prompt_template = (
                system_prompt
                + """
    
            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt
                + """
            
            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    print(f"Here is the prompt used: {prompt}")

    return (
        prompt,
        memory,
    )
