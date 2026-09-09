"""
Wire Ollama chat + embeddings into SIRAT process_biomedical inputs.

Returns token logprobs (for 5-pillar core) and optional embedding vectors for Haqq.
"""



from __future__ import annotations



import numpy as np



from sirat.ollama.ollama_bridge import ollama_chat, ollama_embed





def ollama_generate_for_sirat(

    base_url: str,

    model: str,

    user_message: str,

    system: str | None,

    *,

    use_five_pillar: bool = True,

    timeout_chat_s: float = 180.0,

    timeout_embed_s: float = 120.0,

) -> tuple[str, list[float] | None, np.ndarray | None, np.ndarray | None, str | None]:

    """

    Returns (assistant_text, logprobs_or_none, v_output, v_axiom, embed_error_or_none).

    """

    if not use_five_pillar:

        r = ollama_chat(

            base_url,

            model,

            user_message,

            system=system,

            timeout_s=timeout_chat_s,

            return_logprobs=False,

        )

        return r.message, None, None, None, None



    r = ollama_chat(

        base_url,

        model,

        user_message,

        system=system,

        timeout_s=timeout_chat_s,

        return_logprobs=True,

        top_logprobs=0,

    )

    lps = r.token_logprobs

    logprob_arg: list[float] | None = lps if len(lps) > 0 else None



    v_out: np.ndarray | None = None

    v_ax: np.ndarray | None = None

    err: str | None = None

    try:

        anchor = user_message.strip() or "."

        body = (r.message or "").strip() or "."

        ea = ollama_embed(base_url, model, anchor, timeout_s=timeout_embed_s)

        eo = ollama_embed(base_url, model, body, timeout_s=timeout_embed_s)

        v_ax = np.asarray(ea, dtype=np.float64)

        v_out = np.asarray(eo, dtype=np.float64)

    except Exception as e:

        err = str(e)



    return r.message, logprob_arg, v_out, v_ax, err