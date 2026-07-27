"""Grounded CoachOS assistant: coach-scoped retrieval, citations, and memory."""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, func, select, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Settings(BaseSettings):
    database_url: str
    jwt_secret_key: str
    internal_service_token: str
    openai_api_key: str | None = None
    embedding_dimensions: int = 128
    top_k: int = 6
    similarity_threshold: float = 0.35
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # type: ignore[call-arg]
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "assistant_conversations"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    coach_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    summary: Mapped[str] = mapped_column(Text, default="")
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "assistant_messages"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("assistant_conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Chunk(Base):
    __tablename__ = "assistant_chunks"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    stable_key: Mapped[str] = mapped_column(String(300), unique=True, index=True)
    chunk_type: Mapped[str] = mapped_column(String(64), index=True)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(128))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    organization_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    coach_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), index=True)
    athlete_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), index=True)
    entity_id: Mapped[str] = mapped_column(String(100), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Identity(BaseModel):
    id: UUID
    role: str


def db() -> Session:
    with SessionLocal() as session:
        yield session


def coach(authorization: str = Header()) -> Identity:
    try:
        token = authorization.removeprefix("Bearer ")
        claims = jwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        identity = Identity(id=UUID(str(claims.get("sub") or claims.get("user_id"))), role=claims["role"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(401, "Invalid access token.") from exc
    if identity.role not in {"coach", "admin"}:
        raise HTTPException(403, "Coach access is required.")
    return identity


def embed(value: str) -> list[float]:
    """Deterministic local embedding fallback; production swaps this for OpenAI provider."""
    vector = [0.0] * 128
    for token in value.lower().split():
        vector[hash(token) % len(vector)] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


class IndexRequest(BaseModel):
    entity_type: Literal["athlete_profile", "approved_review", "drill_assignment", "goal", "timeline_event", "practice_session", "insight_summary"]
    entity_id: str
    coach_id: UUID
    athlete_id: UUID | None = None
    content: str = Field(min_length=1, max_length=20_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None
    athlete_id: UUID | None = None


def require_internal(x_service_token: str = Header()) -> None:
    if x_service_token != settings.internal_service_token:
        raise HTTPException(401, "Invalid internal service token.")


def retrieve(session: Session, identity: Identity, question: str, athlete_id: UUID | None) -> list[tuple[Chunk, float]]:
    vector = embed(question)
    distance = Chunk.embedding.cosine_distance(vector)
    statement = select(Chunk, (1 - distance).label("score")).where(
        Chunk.organization_id == identity.id, Chunk.coach_id == identity.id,
    )
    if athlete_id:
        statement = statement.where(Chunk.athlete_id == athlete_id)
    rows = session.execute(statement.order_by(distance).limit(settings.top_k)).all()
    return [(chunk, float(score)) for chunk, score in rows if float(score) >= settings.similarity_threshold]


def answer(question: str, sources: list[tuple[Chunk, float]]) -> str:
    if not sources:
        return "I don’t have enough CoachOS evidence to answer that. Try selecting an athlete or ask about recorded goals, reviews, drills, sessions, or timeline events."
    evidence = " ".join(f"[{index + 1}] {chunk.content}" for index, (chunk, _) in enumerate(sources))
    return f"Based on the retrieved CoachOS records: {evidence[:3500]}"


app = FastAPI(title="CoachOS Assistant Service")


@app.on_event("startup")
def startup() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)


@app.get("/health/ready")
def ready() -> dict[str, str]:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.post("/internal/index", dependencies=[Depends(require_internal)])
def index(payload: IndexRequest, session: Session = Depends(db)) -> dict[str, str]:
    # Coach ID is the current organization boundary until an organization domain exists.
    key = f"{payload.coach_id}:{payload.entity_type}:{payload.entity_id}"
    chunk = session.scalar(select(Chunk).where(Chunk.stable_key == key))
    values = {"chunk_type": payload.entity_type, "content": payload.content, "embedding": embed(payload.content), "metadata_json": payload.metadata, "organization_id": payload.coach_id, "coach_id": payload.coach_id, "athlete_id": payload.athlete_id, "entity_id": payload.entity_id}
    if chunk:
        for field, value in values.items():
            setattr(chunk, field, value)
    else:
        chunk = Chunk(stable_key=key, **values)
        session.add(chunk)
    session.commit()
    return {"status": "indexed", "chunk_id": str(chunk.id)}


@app.post("/api/v1/chat")
def chat(payload: ChatRequest, identity: Identity = Depends(coach), session: Session = Depends(db)) -> dict[str, Any]:
    conversation = session.get(Conversation, payload.conversation_id) if payload.conversation_id else None
    if conversation and (conversation.coach_id != identity.id or conversation.organization_id != identity.id):
        raise HTTPException(404, "Conversation not found.")
    if not conversation:
        conversation = Conversation(coach_id=identity.id, organization_id=identity.id, title=payload.question[:80])
        session.add(conversation)
        session.flush()
    sources = retrieve(session, identity, payload.question, payload.athlete_id)
    response = answer(payload.question, sources)
    citations = [{"chunk_id": str(chunk.id), "entity_type": chunk.chunk_type, "entity_id": chunk.entity_id, "athlete_id": str(chunk.athlete_id) if chunk.athlete_id else None, "confidence": round(score, 3)} for chunk, score in sources]
    session.add_all([Message(conversation_id=conversation.id, role="user", content=payload.question), Message(conversation_id=conversation.id, role="assistant", content=response, citations={"items": citations})])
    session.commit()
    return {"conversation_id": str(conversation.id), "answer": response, "citations": citations, "suggested_follow_ups": ["Show the underlying timeline", "What goals are related to this?"] if sources else []}


@app.post("/api/v1/chat/stream")
def chat_stream(payload: ChatRequest, identity: Identity = Depends(coach), session: Session = Depends(db)) -> StreamingResponse:
    result = chat(payload, identity, session)
    def events() -> Any:
        for token in result["answer"].split(" "):
            yield f"event: token\ndata: {json.dumps({'token': token + ' '})}\n\n"
        yield f"event: citations\ndata: {json.dumps(result['citations'])}\n\n"
        yield "event: done\ndata: {}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/v1/conversations")
def conversations(identity: Identity = Depends(coach), session: Session = Depends(db)) -> list[dict[str, Any]]:
    rows = session.scalars(select(Conversation).where(Conversation.coach_id == identity.id, Conversation.archived_at.is_(None)).order_by(Conversation.updated_at.desc())).all()
    return [{"id": str(row.id), "title": row.title, "updated_at": row.updated_at} for row in rows]


@app.get("/api/v1/conversations/{conversation_id}")
def conversation(conversation_id: UUID, identity: Identity = Depends(coach), session: Session = Depends(db)) -> dict[str, Any]:
    row = session.get(Conversation, conversation_id)
    if not row or row.coach_id != identity.id:
        raise HTTPException(404, "Conversation not found.")
    return {"id": str(row.id), "title": row.title, "messages": [{"role": item.role, "content": item.content, "citations": item.citations} for item in row.messages]}


@app.delete("/api/v1/conversations/{conversation_id}", status_code=204, response_class=Response)
def archive(conversation_id: UUID, identity: Identity = Depends(coach), session: Session = Depends(db)) -> Response:
    row = session.get(Conversation, conversation_id)
    if not row or row.coach_id != identity.id:
        raise HTTPException(404, "Conversation not found.")
    row.archived_at = datetime.now(UTC)
    session.commit()
    return Response(status_code=204)
