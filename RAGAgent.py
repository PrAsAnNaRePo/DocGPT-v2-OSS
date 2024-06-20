from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
import json

class Agent:
    def __init__(
            self,
            system_prompt: str,
            model: str = 'qwen2:7b',
            embed_model: str = 'all-MiniLM-L6-v2',
            doc_store_dir: str = './documents',
            verbose: bool = False
    ) -> None:
        
        self.model = model
        self.doc_store_dir = doc_store_dir
        self.verbose = verbose

        self.core_messages = [
            {
                'role': 'system',
                'content': system_prompt
            }
        ]
        self.options = {
            'temperature': 0.75
        }
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        
        self.embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs=model_kwargs)

        self.vector_store = None
    
    def set_document(self, documents):
        total_content = ''
        num_pages = 0
        for file in documents:
            file_type = file.type
            if file_type == "application/pdf":
                pdf_reader = PdfReader(file)
                content = ''
                for page in pdf_reader.pages:
                    num_pages += 1
                    content += page.extract_text()

            if file_type == "text/plain":
                content = file.read()
                content = content.decode("utf-8")

            total_content += content

        # if num_pages <= 2:
        #     chunk_size = 500
        # elif num_pages <= 3:
        #     chunk_size = 1000
        # elif num_pages <= 5:
        #     chunk_size = 2000
        # elif num_pages <= 10:
        #     chunk_size = 3000
        # else:
        #     chunk_size = 5000
        chunk_size = 1000
        
        if self.verbose:
            print("Chunk size: ", chunk_size)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        texts = text_splitter.split_text(total_content)
        self.vector_store = Chroma.from_texts(texts, self.embeddings).as_retriever()
        if self.verbose:
            print("Vector store updated!!!")

    def generate_related_quires(self, query):
        response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': """You are follow-up query generated. The user will share a query that they are searching in a document (like pdf, docs etc.). Your job is to generate 2 or 3 more queries with same context. Something like follow-up queries (includes what could user will ask after this query relatively) and related queries (related to the query in different tone).
Use following JSON schema for response:
{
    "follow_up_query": [
        "follow_up_query_1",
        "follow_up_query_2",
    ],
    "related_queries": [
        "related_query_1",
        "related_query_2",
    ]
}"""
                    },
                    {
                        'role': 'user',
                        'content': f'Query: {query}'
                    }
                ],
                format='json'
            )['message']['content']
        
        if self.verbose:
            print(response)
        
        response_schema = json.loads(response)
        follow_ups = response_schema['follow_up_query']
        related_queries = response_schema['related_queries']
        
        if self.verbose:
            print("Follow-ups:\n" + str(follow_ups))
            print("Related queries:\n" + str(related_queries))

        return follow_ups, related_queries

    def convert_queries_to_keywords(self, query):
        response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': """Given a question by user, you have to convert it into nice simple form or like keyword. Just convert the question into simpler form that looks easier. Don't include any other text, just the query as response."""
                    },
                    {
                        'role': 'user',
                        'content': f'Question: What all are the related works they considered?'
                    },
                    {
                        'role': 'assistant',
                        'content': 'Related works'
                    },
                    {
                        'role': 'user',
                        'content': query
                    }
                ],
            )['message']['content']
        return response
    
    def __call__(self, query, return_addn_queries=False):

        self.core_messages.append(
            {
                'role': 'user',
                'content': query
            }
        )
        response = ollama.chat(
            model=self.model,
            messages=self.core_messages,
            format='json'
        )['message']['content']
        
        if self.verbose:
            print(response)

        self.core_messages.append(
            {
                'role': 'assistant',
                'content': response
            }
        )
        
        response_schema = json.loads(response)
        check_doc = response_schema['needed_doc_content']

        if check_doc:
            print("ENtereed...")
            if self.vector_store is not None:
                followups, relateds = self.generate_related_quires(query)
                additional_queries = followups + relateds
                additional_queries.append(query)
                if self.verbose:
                    print("Additional queries:\n"+str(additional_queries))
                
                totall_content = ""
                for i in additional_queries:
                    # query_keyword = self.convert_queries_to_keywords(i)
                    # print(">> ", query_keyword)
                    totall_content += str(self.vector_store.get_relevant_documents(i)) + "\n\n# ----------------------------\n\n"
                if self.verbose:
                    print("# ---------------------------------------")
                    print(totall_content)
                    print("# ---------------------------------------")
                    
                self.core_messages.append(
                    {
                        'role': 'user',
                        'content': f"This is document assistant, not user. here is the content: {totall_content}.\nQuestion: {query}"
                    }
                )

            else:
                print("enterrredddd!!!")
                self.core_messages.append(
                    {
                        'role': 'user',
                        'content': "This is document assistant, not user. No documents are uploaded yet!"
                    }
                )

            response = ollama.chat(
                model=self.model,
                messages=self.core_messages,
            )['message']['content']

            if self.verbose:
                print(response)
            
            self.core_messages.append(
                {
                    'role': 'assistant',
                    'content': response
                }
            )

            if return_addn_queries and self.vector_store is not None:
                return response, additional_queries
            return response
        
        else:
            try:
                return response['answer']
            except:
                return response