from __future__ import annotations



import json

import os

import re

from ast import literal_eval

from dataclasses import dataclass

from typing import Any, Callable

from urllib import error, request



from kathnlp.pipelines.corpus import simple_tokenize

from kathnlp.schema import KatharevousaExtensions, UDSentenceAnnotation, UDTokenAnnotation





DEFAULT_PROMPT_TEMPLATE = """You are a linguistic annotator for Katharevousa Greek.

Return ONLY valid JSON with this schema:

{{

  "tokens": [

    {{

      "id": 1,

      "form": "token",

      "lemma": "lemma",

      "upos": "NOUN",

      "xpos": "_",

      "feats": {{"Case": "Nom"}},

      "head": 0,

      "deprel": "root",

      "misc": {{}}

    }}

  ]

}}



Sentence:

{sentence}

"""



STRICT_JSON_PROMPT_TEMPLATE = """You are a strict JSON generator for UD-style token annotations.

Return exactly one valid JSON object and nothing else.

Rules:

- No markdown fences.

- No comments.

- No trailing commas.

- Use double quotes for all keys and string values.

- Output must parse with Python json.loads.

- The object must contain a top-level key "tokens" with an array of token objects.



Sentence:

{sentence}

"""



UD_ANNOTATION_JSON_SCHEMA: dict[str, Any] = {

    "type": "object",

    "additionalProperties": False,

    "properties": {

        "tokens": {

            "type": "array",

            "items": {

                "type": "object",

                "additionalProperties": True,

                "properties": {

                    "id": {"type": "integer"},

                    "form": {"type": "string"},

                    "lemma": {"type": "string"},

                    "upos": {"type": "string"},

                    "xpos": {"type": "string"},

                    "head": {"type": "integer"},

                    "deprel": {"type": "string"},

                    "feats": {"type": "object"},

                    "misc": {"type": "object"},

                    "katharevousa": {"type": "object"},

                },

                "required": ["id", "form"],

            },

        }

    },

    "required": ["tokens"],

}





def _safe_json_loads(payload: str) -> dict[str, Any]:

    def _to_dict(value: Any) -> dict[str, Any]:

        if isinstance(value, dict):

            return value

        if isinstance(value, list):

            return {"tokens": value}

        raise ValueError("Model output is not a JSON object")



    payload = payload.strip()

    if payload.startswith("```"):

        payload = payload.split("\n", 1)[1]

        payload = payload.rsplit("```", 1)[0]



    candidates: list[str] = [payload]

    match = re.search(r"\{.*\}", payload, flags=re.DOTALL)

    if match:

        candidates.append(match.group(0))

    repaired = _repair_json_like_text(payload)

    if repaired and repaired not in candidates:

        candidates.append(repaired)

        repaired_match = re.search(r"\{.*\}", repaired, flags=re.DOTALL)

        if repaired_match:

            candidates.append(repaired_match.group(0))



    for candidate in candidates:

        try:

            return _to_dict(json.loads(candidate))

        except Exception:

            pass

        try:

            return _to_dict(literal_eval(candidate))

        except Exception:

            pass

    raise ValueError("Model output is not parseable JSON")





def _repair_json_like_text(payload: str) -> str:

    text = payload.strip()

    if not text:

        return text

    # Normalize quote variants that often appear in model outputs.

    text = (

        text.replace("“", '"')

        .replace("”", '"')

        .replace("„", '"')

        .replace("’", "'")

        .replace("‘", "'")

    )

    # Remove trailing commas before object/array close.

    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Quote likely bare keys: {key: ...} or ,key: ...

    text = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)', r'\1"\2"\3', text)

    return text





def _extract_openai_text(raw: dict[str, Any]) -> str:

    if isinstance(raw.get("output_text"), str) and raw.get("output_text"):

        return str(raw["output_text"])

    for item in raw.get("output", []):

        if item.get("type") != "message":

            continue

        for chunk in item.get("content", []):

            if chunk.get("type") == "output_text" and chunk.get("text"):

                return str(chunk["text"])

            if chunk.get("type") == "text" and chunk.get("text"):

                return str(chunk["text"])

    return "{}"





def _extract_openai_tool_arguments(raw: dict[str, Any]) -> str | None:

    choices = raw.get("choices", [])

    if not choices:

        return None

    message = choices[0].get("message", {}) or {}

    tool_calls = message.get("tool_calls", []) or []

    for tool_call in tool_calls:

        if tool_call.get("type") != "function":

            continue

        fn = tool_call.get("function", {}) or {}

        args = fn.get("arguments")

        if isinstance(args, str) and args.strip():

            return args

    return None





def _build_sentence_from_payload(

    sentence_id: str,

    text: str,

    payload: dict[str, Any],

    model_name: str,

) -> UDSentenceAnnotation:

    tokens_payload = payload.get("tokens", [])

    tokens: list[UDTokenAnnotation] = []

    for token in tokens_payload:

        token_kath = token.get("katharevousa", {})

        if not isinstance(token_kath, dict):

            token_kath = {}

        token_feats = token.get("feats", {})

        if not isinstance(token_feats, dict):

            token_feats = {}

        token_misc = token.get("misc", {})

        if not isinstance(token_misc, dict):

            token_misc = {}

        ext = KatharevousaExtensions(

            archaic_lexeme_class=(token_kath or {}).get(

                "archaic_lexeme_class"

            ),

            orthography_source=(token_kath or {}).get(

                "orthography_source"

            ),

            legacy_morphology_flags=(token_kath or {}).get(

                "legacy_morphology_flags", []

            ),

            legal_register_markers=(token_kath or {}).get(

                "legal_register_markers", []

            ),

        )

        tokens.append(

            UDTokenAnnotation(

                id=int(token["id"]),

                form=str(token["form"]),

                lemma=str(token.get("lemma", token["form"])),

                upos=str(token.get("upos", "X")),

                xpos=str(token.get("xpos", "_")),

                feats={str(k): str(v) for k, v in (token_feats or {}).items()},

                head=int(token.get("head", 0)),

                deprel=str(token.get("deprel", "dep")),

                misc={str(k): str(v) for k, v in (token_misc or {}).items()},

                katharevousa=ext,

            )

        )

    if not tokens:

        raise ValueError(f"{model_name} produced empty token list")

    return UDSentenceAnnotation(

        sentence_id=sentence_id,

        text=text,

        tokens=tokens,

        metadata={"annotator": model_name},

    )





def _heuristic_annotation(

    sentence_id: str,

    text: str,

    model_name: str,

    variant: str = "base",

) -> UDSentenceAnnotation:

    token_texts = simple_tokenize(text)

    tokens: list[UDTokenAnnotation] = []

    root_id = 1 if token_texts else 0

    if variant == "late_root" and len(token_texts) > 2:

        root_id = 2

    for idx, tok in enumerate(token_texts, start=1):

        if tok.isalpha():

            if variant == "proper_bias" and tok[:1].isupper():

                upos = "PROPN"

            else:

                upos = "NOUN" if idx != root_id else "VERB"

        elif tok.isnumeric():

            upos = "NUM"

        else:

            upos = "PUNCT"

        tokens.append(

            UDTokenAnnotation(

                id=idx,

                form=tok,

                lemma=tok.lower(),

                upos=upos,

                head=0 if idx == root_id else root_id,

                deprel="root" if idx == root_id else "dep",

                feats={},

                katharevousa=KatharevousaExtensions(

                    orthography_source="polytonic_original",

                ),

            )

        )

    return UDSentenceAnnotation(

        sentence_id=sentence_id,

        text=text,

        tokens=tokens,

        metadata={"annotator": model_name, "mode": "heuristic_fallback"},

    )





@dataclass

class LLMAnnotator:

    model_name: str

    annotate_fn: Callable[[str, str], UDSentenceAnnotation]



    def annotate(self, sentence_id: str, sentence_text: str) -> UDSentenceAnnotation:

        return self.annotate_fn(sentence_id, sentence_text)





def build_openai_annotator(

    strict_json: bool = False,

    model_override: str | None = None,

    max_completion_tokens_override: int | None = None,

) -> LLMAnnotator:

    api_key = os.getenv("OPENAI_API_KEY")

    model = model_override or os.getenv("OPENAI_MODEL", "gpt-5.5")

    max_completion_tokens = max_completion_tokens_override or int(

        os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "2500")

    )

    prompt_template = (

        STRICT_JSON_PROMPT_TEMPLATE if strict_json else DEFAULT_PROMPT_TEMPLATE

    )

    structured_mode = os.getenv("OPENAI_STRUCTURED_MODE", "0") == "1"

    if not api_key:

        return LLMAnnotator(

            model_name="openai_heuristic",

            annotate_fn=lambda sid, text: _heuristic_annotation(

                sid, text, "openai_heuristic", variant="base"

            ),

        )



    def _annotate(sentence_id: str, text: str) -> UDSentenceAnnotation:

        def _call_openai(

            call_model: str,

            mode: str,

        ) -> dict[str, Any]:

            payload: dict[str, Any] = {

                "model": call_model,

                "messages": [

                    {

                        "role": "user",

                        "content": prompt_template.format(sentence=text),

                    }

                ],

                "max_completion_tokens": max_completion_tokens,

            }

            if mode == "function_call":

                payload["tools"] = [

                    {

                        "type": "function",

                        "function": {

                            "name": "annotate_ud_sentence",

                            "description": "Return UD token annotation as JSON.",

                            "parameters": UD_ANNOTATION_JSON_SCHEMA,

                        },

                    }

                ]

                payload["tool_choice"] = {

                    "type": "function",

                    "function": {"name": "annotate_ud_sentence"},

                }

            elif mode == "json_schema":

                payload["response_format"] = {

                    "type": "json_schema",

                    "json_schema": {

                        "name": "ud_annotation",

                        "strict": True,

                        "schema": UD_ANNOTATION_JSON_SCHEMA,

                    },

                }

            else:

                payload["response_format"] = {"type": "json_object"}

            req = request.Request(

                "https://api.openai.com/v1/chat/completions",

                data=json.dumps(payload).encode("utf-8"),

                headers={

                    "Authorization": f"Bearer {api_key}",

                    "Content-Type": "application/json",

                },

                method="POST",

            )

            with request.urlopen(req, timeout=240) as response:

                return json.loads(response.read().decode("utf-8"))



        used_model = model

        mode_order = (

            ("function_call", "json_schema", "json_object")

            if structured_mode

            else ("json_object",)

        )

        raw: dict[str, Any] | None = None

        last_exc: Exception | None = None

        for mode in mode_order:

            try:

                raw = _call_openai(used_model, mode)

                break

            except error.HTTPError as exc:

                last_exc = exc

                # Some models/tiers may reject advanced response modes.

                if exc.code in (400, 404):

                    continue

                raise

        if raw is None:

            if isinstance(last_exc, error.HTTPError) and last_exc.code == 404 and used_model != "gpt-5.5":

                used_model = "gpt-5.5"

                for mode in mode_order:

                    try:

                        raw = _call_openai(used_model, mode)

                        break

                    except error.HTTPError as exc:

                        last_exc = exc

                        if exc.code in (400, 404):

                            continue

                        raise

            if raw is None and last_exc is not None:

                raise last_exc

            if raw is None:

                raise ValueError("OpenAI response unavailable")

        tool_args = _extract_openai_tool_arguments(raw)

        text_output = (

            tool_args

            or raw.get("choices", [{}])[0]

            .get("message", {})

            .get("content", _extract_openai_text(raw))

        )

        annotation = _build_sentence_from_payload(

            sentence_id=sentence_id,

            text=text,

            payload=_safe_json_loads(text_output),

            model_name=used_model,

        )

        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}

        if isinstance(usage, dict):

            annotation.metadata["prompt_tokens"] = str(usage.get("prompt_tokens", 0))

            annotation.metadata["completion_tokens"] = str(

                usage.get("completion_tokens", 0)

            )

            annotation.metadata["total_tokens"] = str(usage.get("total_tokens", 0))

        if strict_json:

            annotation.metadata["mode"] = "strict_json"

        return annotation



    return LLMAnnotator(model_name=model, annotate_fn=_annotate)





def build_anthropic_annotator() -> LLMAnnotator:

    api_key = os.getenv("ANTHROPIC_API_KEY")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    if not api_key:

        return LLMAnnotator(

            model_name="anthropic_heuristic",

            annotate_fn=lambda sid, text: _heuristic_annotation(

                sid, text, "anthropic_heuristic", variant="proper_bias"

            ),

        )



    def _annotate(sentence_id: str, text: str) -> UDSentenceAnnotation:

        payload = {

            "model": model,

            "max_tokens": 8192,

            "messages": [

                {

                    "role": "user",

                    "content": DEFAULT_PROMPT_TEMPLATE.format(sentence=text),

                }

            ],

        }

        req = request.Request(

            "https://api.anthropic.com/v1/messages",

            data=json.dumps(payload).encode("utf-8"),

            headers={

                "x-api-key": api_key,

                "anthropic-version": "2023-06-01",

                "content-type": "application/json",

            },

            method="POST",

        )

        with request.urlopen(req, timeout=90) as response:

            raw = json.loads(response.read().decode("utf-8"))

        text_output = raw.get("content", [{}])[0].get("text", "{}")

        return _build_sentence_from_payload(

            sentence_id=sentence_id,

            text=text,

            payload=_safe_json_loads(text_output),

            model_name=model,

        )



    return LLMAnnotator(model_name=model, annotate_fn=_annotate)





def build_gemini_annotator() -> LLMAnnotator:

    api_key = os.getenv("GEMINI_API_KEY")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    if not api_key:

        return LLMAnnotator(

            model_name="gemini_heuristic",

            annotate_fn=lambda sid, text: _heuristic_annotation(

                sid, text, "gemini_heuristic", variant="late_root"

            ),

        )



    def _annotate(sentence_id: str, text: str) -> UDSentenceAnnotation:

        url = (

            "https://generativelanguage.googleapis.com/v1beta/models/"

            f"{model}:generateContent?key={api_key}"

        )

        payload = {

            "contents": [{"parts": [{"text": DEFAULT_PROMPT_TEMPLATE.format(sentence=text)}]}]

        }

        req = request.Request(

            url,

            data=json.dumps(payload).encode("utf-8"),

            headers={"content-type": "application/json"},

            method="POST",

        )

        with request.urlopen(req, timeout=90) as response:

            raw = json.loads(response.read().decode("utf-8"))

        text_output = (

            raw.get("candidates", [{}])[0]

            .get("content", {})

            .get("parts", [{}])[0]

            .get("text", "{}")

        )

        return _build_sentence_from_payload(

            sentence_id=sentence_id,

            text=text,

            payload=_safe_json_loads(text_output),

            model_name=model,

        )



    return LLMAnnotator(model_name=model, annotate_fn=_annotate)

