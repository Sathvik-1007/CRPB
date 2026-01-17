from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fs import atomic_write_json, lock_file, read_json


def _parse_provider_preference(provider_preference: Optional[List[str]]) -> List[str]:
    if provider_preference is None:
        raw = os.environ.get("CRPB_TASK_EMBED_PROVIDER_PREFERENCE")
        if isinstance(raw, str) and raw.strip():
            provider_preference = [p.strip() for p in raw.split(",") if p.strip()]
        else:
            provider_preference = []
    pref: List[str] = []
    seen: set[str] = set()
    for p in provider_preference:
        pn = str(p or "").strip().lower()
        if not pn or pn in seen:
            continue
        seen.add(pn)
        pref.append(pn)
    return pref


def _split_text_chunks(text: str, max_chars: int) -> List[str]:
    t = str(text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return []
    if max_chars <= 0:
        return [t]
    if len(t) <= int(max_chars):
        return [t]
    out: List[str] = []
    start = 0
    n = len(t)
    step = int(max_chars)
    while start < n:
        chunk = t[start : min(n, start + step)].strip()
        if chunk:
            out.append(chunk)
        start += step
    return out


def normalize_task_embedding_fields(*, title: str, summary: str) -> Tuple[str, str, str]:
    title2 = str(title or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    summary2 = str(summary or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = (title2 + "\n" + summary2).strip()
    if not text:
        text = title2.strip()
    return title2, summary2, text


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _best_effort_extract_embeddings(payload: Any) -> List[List[float]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            out: List[List[float]] = []
            for it in data:
                if isinstance(it, dict):
                    emb = it.get("embedding")
                    if isinstance(emb, list):
                        out.append([float(x) for x in emb])
            if out:
                return out
        embs = payload.get("embeddings")
        if isinstance(embs, list):
            if embs and isinstance(embs[0], list):
                return [[float(x) for x in e] for e in embs]
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return [[float(x) for x in e] for e in payload]
    return []


class VoyageEmbedder:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        input_type: str = "document",
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.timeout_s = timeout_s

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(
                {"input": texts, "model": self.model, "input_type": self.input_type}
            ).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
        embs = _best_effort_extract_embeddings(payload)
        if len(embs) != len(texts):
            raise RuntimeError("voyage_embedding_count_mismatch")
        return embs


class OpenAIEmbedder:
    def __init__(self, *, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=self.timeout_s)
        resp = client.embeddings.create(model=self.model, input=texts)
        out: List[List[float]] = []
        for it in getattr(resp, "data", []) or []:
            emb = getattr(it, "embedding", None)
            if isinstance(emb, list):
                out.append([float(x) for x in emb])
        if len(out) != len(texts):
            raise RuntimeError("openai_embedding_count_mismatch")
        return out


def make_embedder_from_env(
    *,
    provider: str,
    input_type: str = "document",
    voyage_model: Optional[str] = None,
    openai_model: Optional[str] = None,
) -> Any:
    """Create an embedder instance using environment configuration.

    Contract: no hidden model defaults. If a provider is requested but the corresponding
    model is not provided (via args or env), returns None.
    """

    p = str(provider or "").strip().lower()
    if p == "voyage":
        voyage_key = os.environ.get("VOYAGE_API_KEY")
        model = str(voyage_model or os.environ.get("CRPB_VOYAGE_EMBED_MODEL") or "").strip()
        if not voyage_key or not model:
            return None
        return VoyageEmbedder(
            api_key=voyage_key,
            model=model,
            input_type=str(input_type or "document"),
            timeout_s=float(os.environ.get("CRPB_VOYAGE_EMBED_TIMEOUT_S", "30") or 30.0),
        )

    if p == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY")
        model = str(openai_model or os.environ.get("CRPB_OPENAI_EMBED_MODEL") or "").strip()
        if not openai_key or not model:
            return None
        return OpenAIEmbedder(
            api_key=openai_key,
            model=model,
            timeout_s=float(os.environ.get("CRPB_OPENAI_EMBED_TIMEOUT_S", "30") or 30.0),
        )

    return None


@dataclass
class TaskEmbeddingItem:
    id: str
    path: str
    title: str
    summary: str
    text_hash: str
    vectors: Dict[str, List[float]]


class TaskEmbeddingIndex:
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self._obj: Dict[str, Any] = {}
        self._items: List[TaskEmbeddingItem] = []

    def load(self) -> None:
        obj = read_json(self.path, default={})
        if not isinstance(obj, dict):
            obj = {}
        items_raw = obj.get("items")
        items: List[TaskEmbeddingItem] = []
        if isinstance(items_raw, list):
            for it in items_raw:
                if not isinstance(it, dict):
                    continue
                rid = str(it.get("id") or "")
                if not rid:
                    continue
                vectors = it.get("vectors")
                if not isinstance(vectors, dict):
                    vectors = {}
                vv: Dict[str, List[float]] = {}
                for k, v in vectors.items():
                    if isinstance(k, str) and isinstance(v, list):
                        try:
                            vv[k] = [float(x) for x in v]
                        except Exception:
                            continue
                items.append(
                    TaskEmbeddingItem(
                        id=rid,
                        path=str(it.get("path") or ""),
                        title=str(it.get("title") or ""),
                        summary=str(it.get("summary") or ""),
                        text_hash=str(it.get("text_hash") or ""),
                        vectors=vv,
                    )
                )
        self._obj = obj
        self._items = items

    def save(self) -> None:
        obj = dict(self._obj or {})
        obj.setdefault("version", 1)
        obj.setdefault("created_at", time.time())
        obj["updated_at"] = time.time()
        obj["items"] = [
            {
                "id": it.id,
                "path": it.path,
                "title": it.title,
                "summary": it.summary,
                "text_hash": it.text_hash,
                "vectors": it.vectors,
            }
            for it in self._items
        ]
        atomic_write_json(self.path, obj)
        self._obj = obj

    def upsert(self, item: TaskEmbeddingItem) -> None:
        if not item.id:
            return
        for i in range(len(self._items)):
            if self._items[i].id == item.id:
                self._items[i] = item
                return
        self._items.append(item)

    def get_vector(
        self, item: TaskEmbeddingItem, provider_preference: List[str]
    ) -> Optional[List[float]]:
        for p in provider_preference:
            v = item.vectors.get(p)
            if isinstance(v, list) and v:
                return v
        return None

    def query_similar(
        self,
        *,
        vector: List[float],
        top_k: int = 8,
        min_score: float = 0.0,
        provider_preference: Optional[List[str]] = None,
    ) -> List[Tuple[TaskEmbeddingItem, float]]:
        pref = _parse_provider_preference(provider_preference)
        if not pref:
            return []
        scored: List[Tuple[TaskEmbeddingItem, float]] = []
        for it in self._items:
            v = self.get_vector(it, pref)
            if v is None:
                continue
            s = _cosine_similarity(vector, v)
            if s >= float(min_score):
                scored.append((it, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: max(0, int(top_k))]


def build_task_embedding_index(
    *,
    index_path: Path,
    items: List[Dict[str, Any]],
    provider_preference: Optional[List[str]] = None,
    voyage_model: Optional[str] = None,
    openai_model: Optional[str] = None,
) -> Dict[str, Any]:
    pref = _parse_provider_preference(provider_preference)

    voyage_key = os.environ.get("VOYAGE_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    voyage_model_use = str(voyage_model or os.environ.get("CRPB_VOYAGE_EMBED_MODEL") or "").strip()
    openai_model_use = str(openai_model or os.environ.get("CRPB_OPENAI_EMBED_MODEL") or "").strip()

    try:
        batch_size = int(os.environ.get("CRPB_TASK_EMBED_BATCH_SIZE", "64"))
    except Exception:
        batch_size = 64
    batch_size = max(1, min(512, int(batch_size)))
    try:
        retries = int(os.environ.get("CRPB_TASK_EMBED_RETRIES", "2"))
    except Exception:
        retries = 2
    retries = max(0, min(8, int(retries)))
    try:
        backoff_s = float(os.environ.get("CRPB_TASK_EMBED_RETRY_BACKOFF", "1.0"))
    except Exception:
        backoff_s = 1.0
    backoff_s = max(0.0, min(30.0, float(backoff_s)))
    try:
        voyage_timeout_s = float(os.environ.get("CRPB_VOYAGE_EMBED_TIMEOUT_S", "30"))
    except Exception:
        voyage_timeout_s = 30.0
    try:
        openai_timeout_s = float(os.environ.get("CRPB_OPENAI_EMBED_TIMEOUT_S", "30"))
    except Exception:
        openai_timeout_s = 30.0

    embed_voyage = None
    if voyage_key and "voyage" in pref:
        if voyage_model_use:
            embed_voyage = VoyageEmbedder(
                api_key=voyage_key,
                model=voyage_model_use,
                timeout_s=voyage_timeout_s,
            )

    embed_openai = None
    if openai_key and "openai" in pref:
        if openai_model_use:
            embed_openai = OpenAIEmbedder(
                api_key=openai_key,
                model=openai_model_use,
                timeout_s=openai_timeout_s,
            )

    idx = TaskEmbeddingIndex(path=index_path)
    with lock_file(index_path):
        idx.load()

        try:
            max_text_chars = int(os.environ.get("CRPB_TASK_EMBED_MAX_TEXT_CHARS", "4000"))
        except Exception:
            max_text_chars = 4000
        max_text_chars = max(200, min(20000, int(max_text_chars)))

        existing_by_id: Dict[str, TaskEmbeddingItem] = {}
        try:
            for it in getattr(idx, "_items", []) or []:
                if isinstance(getattr(it, "id", None), str) and it.id:
                    existing_by_id[it.id] = it
        except Exception:
            existing_by_id = {}

        def _retry(fn, texts: List[str]) -> List[List[float]]:
            last: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(texts)
                except Exception as e:
                    last = e
                    if attempt >= retries:
                        break
                    try:
                        time.sleep(backoff_s * (2.0 ** float(attempt)))
                    except Exception:
                        pass
            raise RuntimeError(str(last) if last else "embed_failed")

        def _batched(fn, texts: List[str]) -> List[List[float]]:
            out: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                if not batch:
                    continue
                out.extend(_retry(fn, batch))
            return out

        norm_items: List[Dict[str, Any]] = []
        needs_voyage: List[bool] = []
        needs_openai: List[bool] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tid = str(it.get("id") or "")
            if not tid:
                continue
            title_raw = str(it.get("title") or "")
            summary_raw = str(it.get("summary") or "")
            title, summary, text = normalize_task_embedding_fields(
                title=title_raw, summary=summary_raw
            )
            path = str(it.get("path") or "")
            if not text:
                continue
            text_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
            ex = existing_by_id.get(tid)
            ex_vectors = dict(ex.vectors) if ex is not None else {}

            need_v = bool(embed_voyage is not None)
            if need_v and ex is not None and str(ex.text_hash or "") == text_hash:
                v0 = ex_vectors.get("voyage")
                if isinstance(v0, list) and v0:
                    need_v = False

            need_o = bool(embed_openai is not None)
            if need_o and ex is not None and str(ex.text_hash or "") == text_hash:
                v0 = ex_vectors.get("openai")
                if isinstance(v0, list) and v0:
                    need_o = False

            norm_items.append(
                {
                    "id": tid,
                    "title": title,
                    "summary": summary,
                    "path": path,
                    "text": text,
                    "text_hash": text_hash,
                    "vectors": ex_vectors,
                }
            )
            needs_voyage.append(need_v)
            needs_openai.append(need_o)

        voyage_embs: List[List[float]] = []
        openai_embs: List[List[float]] = []
        chunk_owner: List[int] = []
        chunk_owner2: List[int] = []
        if embed_voyage is not None and any(needs_voyage):
            todo_texts: List[str] = []
            for i in range(len(norm_items)):
                if not needs_voyage[i]:
                    continue
                for ch in _split_text_chunks(str(norm_items[i]["text"] or ""), max_text_chars):
                    todo_texts.append(ch)
                    chunk_owner.append(i)
            if todo_texts:
                voyage_embs = _batched(embed_voyage.embed_texts, todo_texts)
        if embed_openai is not None and any(needs_openai):
            todo_texts2: List[str] = []
            for i in range(len(norm_items)):
                if not needs_openai[i]:
                    continue
                for ch in _split_text_chunks(str(norm_items[i]["text"] or ""), max_text_chars):
                    todo_texts2.append(ch)
                    chunk_owner2.append(i)
            if todo_texts2:
                openai_embs = _batched(embed_openai.embed_texts, todo_texts2)

        def _avg_vec(vs: List[List[float]]) -> Optional[List[float]]:
            if not vs:
                return None
            base = vs[0]
            if not isinstance(base, list) or not base:
                return None
            dim = len(base)
            acc = [0.0] * dim
            cnt = 0
            for v in vs:
                if not isinstance(v, list) or len(v) != dim:
                    continue
                for j in range(dim):
                    acc[j] += float(v[j])
                cnt += 1
            if cnt <= 0:
                return None
            return [float(x) / float(cnt) for x in acc]

        voyage_by_item: Dict[int, List[List[float]]] = {}
        openai_by_item: Dict[int, List[List[float]]] = {}

        if embed_voyage is not None and voyage_embs:
            if len(chunk_owner) == len(voyage_embs):
                for owner, vec in zip(chunk_owner, voyage_embs, strict=True):
                    voyage_by_item.setdefault(int(owner), []).append(vec)
            else:
                # Defensive fallback: map sequentially to each item needing embeddings.
                vi = 0
                for i in range(len(norm_items)):
                    if needs_voyage[i] and vi < len(voyage_embs):
                        voyage_by_item.setdefault(i, []).append(voyage_embs[vi])
                        vi += 1

        if embed_openai is not None and openai_embs:
            if len(chunk_owner2) == len(openai_embs):
                for owner, vec in zip(chunk_owner2, openai_embs, strict=True):
                    openai_by_item.setdefault(int(owner), []).append(vec)
            else:
                # Defensive fallback: map sequentially to each item needing embeddings.
                oi = 0
                for i in range(len(norm_items)):
                    if needs_openai[i] and oi < len(openai_embs):
                        openai_by_item.setdefault(i, []).append(openai_embs[oi])
                        oi += 1

        for i in range(len(norm_items)):
            ni = norm_items[i]
            vectors: Dict[str, List[float]] = dict(ni.get("vectors") or {})
            if embed_voyage is not None and needs_voyage[i]:
                av = _avg_vec(voyage_by_item.get(i, []) or [])
                if isinstance(av, list) and av:
                    vectors["voyage"] = av
            if embed_openai is not None and needs_openai[i]:
                ao = _avg_vec(openai_by_item.get(i, []) or [])
                if isinstance(ao, list) and ao:
                    vectors["openai"] = ao
            idx.upsert(
                TaskEmbeddingItem(
                    id=ni["id"],
                    path=ni.get("path") or "",
                    title=ni.get("title") or "",
                    summary=ni.get("summary") or "",
                    text_hash=str(ni.get("text_hash") or ""),
                    vectors=vectors,
                )
            )

        idx.save()

    return {
        "ok": True,
        "index_path": str(index_path),
        "providers": [
            p
            for p in pref
            if (
                (p == "voyage" and embed_voyage is not None)
                or (p == "openai" and embed_openai is not None)
            )
        ],
        "items": len(items),
    }
