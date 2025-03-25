import os
import traceback
import ollama
import tenacity
from wrapt_timeout_decorator import timeout
from ollama import Client


# Set Ollama host
os.environ["OLLAMA_HOST"] = "http://localhost:11434"



@timeout(dec_timeout=30, use_signals=False)
def connect_ollama(engine, messages, temperature, max_tokens,
                   top_p):
    try:
        return ollama.chat(
            model=engine,
            messages=messages
        )
    except Exception as e:
        print(f'[ERROR] Ollama model error: {e}')
        raise e


class GPT_Chat:
    def __init__(self, engine, stop=None, max_tokens=1000, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0):
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def get_response(self, prompt, messages=None, max_retry=5, verbose=False):
        conn_success, llm_output = False, ''
        if messages is None:
            messages = [{'role': 'user', 'content': prompt}]

        try:
            r = tenacity.Retrying(
                stop=tenacity.stop_after_attempt(max_retry),
                wait=tenacity.wait_fixed(1.5),
                reraise=True
            )
            response = r.__call__(
                connect_ollama,
                engine=self.engine,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p)
            llm_output = response["message"]["content"]
            if verbose:
                print(f'[INFO] Connection success')
            conn_success = True
        except Exception:
            print(traceback.format_exc())

        return conn_success, llm_output


def main():
    llm_gpt = GPT_Chat(engine='gemma3:12b')
    _, llm_output = llm_gpt.get_response('Do you know about PDDL language for planning?')
    print(llm_output)


if __name__ == '__main__':
    main()
