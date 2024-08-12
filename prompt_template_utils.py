"""
This file implements prompt template for llama based models. 
Modify the prompt template based on the model you select. 
This seems to have significant impact on the output of the LLM.
"""

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# this is specific to Llama-2.


# system_prompt = """Przeanalizuj załączony plik tekstowy, który zawiera dane JSON opisujące strukturę miar i wymiarów związanych z kostką OLAP. Twoim zadaniem jest zwrócenie obiektu JSON zawierającego 4 listy stringów przypisanych do następujących pól: values, rows, columns, and filters. Listy te są potrzebne do wygenerowania raportu w programie Power Query. Za każdym razem dobrze wybierz odpowiednie wartości przypisane do uniqueName, analizując plik tekstowy. W każdej liście umieszczaj wyłącznie elementy reprezentowane przez klucz uniqueName z pliku. Zasady jak masz odpowiedzieć: 1. Lista values może zawierać wyłącznie miary i nie może być pusta. 2. Listy rows, columns i filters mogą zawierać wyłącznie wymiary i mogą być puste. 3. Elementy muszą pochodzić z załączonego pliku. Jeśli nie ma odpowiednich elementów, zwróć puste listy. 4. Nie dodawaj żadnego dodatkowego tekstu ani komentarzy. 5. Pamiętaj żeby wybierać tylko wartości z uniqueName, to znaczy że nie możesz wpisywać żadnych wartości typu WYM_NAZWA tylko samą NAZWA a następnie wpisujesz je do tablic 6. Twoja odpowiedź powinna być wyłącznie obiektem JSON o następującej strukturze z odpowiednimi wartościami pasującymi do utworzenia raportu zgodnego z pytaniem, przykładowo: values: [], rows: [], columns: [], filters: [], Pytanie jest następujące: """

system_prompt = """Plik powinien być związany z kostką OLAP i służy do opisu struktury miar i wymiarów. Przeanalizuj plik i odpowiedz na to pytanie w formie obiektu JSON, zawierającego 4 listy przypisane do następujących pól (values, rows, columns and filters). Pamiętaj że ma być to prawidłowy obiekt typu JSON bez opisu i znaków escape. Tak abym łatwo mógł sparsować JSONa, Do list mogą trafiać jedynie elementy reprezentowane przez uniqueName. Lista values może zawierać wyłącznie miary, natomiast pozostałe listy tylko wymiary. Elementy muszą pochodzić z załączonego plku. Jeśli w pliku nie ma elementów, które by odpowiadały wymaganiom zapytania, nie dodawaj nic od siebie, możesz zwrócić puste listy. Poza tym obiektem z 4 listami nie generuj ŻADNEGO dodatkowego tekstu, to bardzo istotne. Całość odpowiedzi ma składać się tylko z tego obiektu w formacie JSON. Nie opisuj swojej odpowiedzi i nie dodawaj żadnych komentarzy dotyczących zawartości list. Twoja odpowiedź powinna być wyłącznie obiektem JSON o następującej strukturze z odpowiednimi wartościami pasującymi do utworzenia raportu zgodnego z pytaniem, przykładowo: values: [], rows: [], columns: [], filters: [], Zapytanie jest następujące:"""

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
