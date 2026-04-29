import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS     = 180
TEMPERATURE        = 0.2
TOP_P              = 0.80
REPETITION_PENALTY = 1.3

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"


class GymGenerator:

    def __init__(self) -> None:
        print(f"[generator] Loading: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE_STR == "cuda" else torch.float32,
            device_map=DEVICE_STR,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
        )
        print("[generator] Ready ✅")

    @staticmethod
    def build_prompt(query: str, context_docs: list[dict]) -> str:
        # KEY FIX: No "Question:" / "Answer:" labels in prompt at all
        # TinyLlama mirrors whatever format it sees — remove all labels
        facts = "\n".join(
            f"- {doc['answer'].strip()}"
            for doc in context_docs
        )

        system_msg = (
            "You are a Gym & Fitness Assistant. Only answer gym/fitness topics.\n\n"

            "STRICT OUTPUT FORMAT:\n"
            "Write 2-4 sentences of direct fitness advice.\n"
            "Nothing else. No labels. No headers. No questions.\n\n"

            "FORBIDDEN — never output any of these:\n"
            "- 'Question:' or 'Answer:' or 'Q:' or 'A:' labels\n"
            "- The user's question repeated back\n"
            "- Feedback text like 'thank you for feedback' or 'here is an updated version'\n"
            "- Meta phrases like 'based on', 'the context says', 'according to'\n"
            "- Openers like 'Sure', 'Certainly', 'Great question', 'I would'\n\n"

            "IF NOT A FITNESS QUESTION: reply only with:\n"
            "I can only answer gym and fitness questions.\n\n"

            f"FITNESS KNOWLEDGE:\n{facts}\n\n"

            "REPLY WITH FITNESS ADVICE ONLY (2-4 sentences, start immediately):"
        )

        return (
            f"<|system|>\n{system_msg}</s>\n"
            f"<|user|>\n{query}</s>\n"
            f"<|assistant|>\n"
        )

    @staticmethod
    def clean_answer(raw: str, query: str) -> str:
        """Remove all artifacts TinyLlama adds to responses."""
        text = raw.strip()

        # Cut at EOS tokens
        for tok in ["</s>", "<|endoftext|>", "<|end|>", "<|user|>", "<|system|>"]:
            text = text.split(tok)[0].strip()

        # Remove "Question: ... Answer: ..." pattern (most common artifact)
        text = re.sub(
            r'(?i)question\s*:.*?answer\s*:\s*["\']?',
            '',
            text,
            flags=re.DOTALL
        ).strip()

        # Remove standalone Q/A labels at start of lines
        text = re.sub(r'(?im)^(question|answer|q|a)\s*:\s*', '', text).strip()

        # Remove feedback/meta sentences
        meta_patterns = [
            r"thank you for (providing|your) feedback[^.!?]*[.!?]?",
            r"here('s| is) an? updated version[^.!?]*[.!?]?",
            r"incorporating your suggestions[^.!?]*[.!?]?",
            r"based on your feedback[^.!?]*[.!?]?",
            r"hope this helps[^.!?]*[.!?]?",
            r"feel free to ask[^.!?]*[.!?]?",
            r"let me know if[^.!?]*[.!?]?",
            r"the context (says|mentions|states)[^.!?]*[.!?]?",
            r"according to the (context|facts|information)[^.!?]*[.!?]?",
            r"based on the (context|facts|retrieved)[^.!?]*[.!?]?",
            r"here is my (response|answer|updated)[^.!?]*[.!?]?",
            r"i('ve| have) updated[^.!?]*[.!?]?",
        ]
        for pattern in meta_patterns:
            text = re.sub(r'(?i)' + pattern, '', text).strip()

        # Remove query echoed back (if first sentence overlaps >60% with query)
        q_words = set(query.lower().split())
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        filtered = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            overlap = len(q_words & sent_words) / max(len(q_words), 1)
            if overlap < 0.6:
                filtered.append(sent)
        if filtered:
            text = ' '.join(filtered).strip()

        # Remove bad opener phrases → drop first sentence
        bad_openers = [
            "the context", "the retrieved", "according to the", "based on the",
            "the facts", "the information provided", "i can provide",
            "sure,", "certainly,", "of course,", "yes, i can",
            "as per the", "as mentioned", "i would", "i'll ",
            "i am happy", "great question", "good question",
        ]
        low = text.lower()
        for phrase in bad_openers:
            if low.startswith(phrase):
                parts = text.split(".")
                if len(parts) > 1:
                    text = ".".join(parts[1:]).strip()
                break

        # Remove surrounding quotes
        text = text.strip('"').strip("'").strip()

        # Trim to max 4 sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) > 4:
            text = ' '.join(sentences[:4])

        # Ensure proper ending
        if text and text[-1] not in '.!?':
            text += '.'

        return text

    def generate(self, query: str, context_docs: list[dict]) -> str:
        prompt = self.build_prompt(query, context_docs)
        print(f"[generator] Generating for: '{query}'")

        outputs   = self.pipe(prompt)
        full_text = outputs[0]["generated_text"]
        raw       = full_text.split("<|assistant|>")[-1]
        answer    = self.clean_answer(raw, query)

        print(f"[generator] Result: {answer[:100]}")

        fallback = "Please ask a gym or fitness question and I will help you! 💪"
        return answer if len(answer) > 15 else fallback


_generator_instance: GymGenerator | None = None

def get_generator() -> GymGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = GymGenerator()
    return _generator_instance