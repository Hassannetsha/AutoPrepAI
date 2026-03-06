import json
import uuid

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.ml_service import MLPipelineService
from backend.models import Conversation, ConversationMessage
from backend.schemas import ChatResponse, ConversationOut


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AutoPrepAI Backend", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str | None = Form(default=None),
    mode: str = Form(default="chat"),
    selected_intents: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    dataset: UploadFile | None = File(default=None),
    db: Session = Depends(get_db),
):
    parsed_selected_intents: list[str] = []
    if selected_intents:
        try:
            loaded = json.loads(selected_intents)
            if isinstance(loaded, list):
                parsed_selected_intents = [str(item) for item in loaded]
            else:
                raise ValueError("selected_intents must be a JSON array")
        except (json.JSONDecodeError, ValueError):
            parsed_selected_intents = [item.strip() for item in selected_intents.split(",") if item.strip()]

    conversation = None
    if conversation_id:
        try:
            conversation_uuid = uuid.UUID(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid conversation_id") from exc
        conversation = db.get(Conversation, conversation_uuid)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = Conversation()
        db.add(conversation)
        db.flush()

    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset is required")

    file_bytes = await dataset.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty")
    try:
        dataset_df = MLPipelineService.dataframe_from_upload(file_bytes, dataset.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {exc}") from exc

    try:
        result = MLPipelineService.process_message(
            user_message=message or "",
            dataset_df=dataset_df,
            mode=mode,
            selected_intents=parsed_selected_intents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_file = result.get("output_file")
    if output_file:
        result["download_url"] = f"/download/{output_file}"

    assistant_message = "Processing completed successfully."
    stored_message = (message or "").strip() or f"[{mode}]"

    db.add(
        ConversationMessage(
            conversation_id=conversation.id,
            role="user",
            content=stored_message,
            payload=None,
        )
    )
    db.add(
        ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_message,
            payload=json.loads(json.dumps(result, default=str)),
        )
    )
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        assistant_message=assistant_message,
        result=result,
    )


@app.get("/download/{filename}")
def download_processed_file(filename: str):
    try:
        file_path = MLPipelineService.get_output_file_path(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc

    return FileResponse(path=file_path, filename=file_path.name, media_type="text/csv")


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid conversation_id") from exc

    conversation = db.get(Conversation, conversation_uuid)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation
