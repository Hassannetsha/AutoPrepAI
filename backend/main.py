import json
import uuid
import time

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from auth.dependencies import get_current_user
from backend.database import Base, engine, ensure_auth_columns, ensure_conversation_columns, get_db
from backend.ml_service import MLPipelineService
from backend.models import Conversation, ConversationMessage, User
from backend.schemas import ChatResponse, ConversationOut
from auth import signup, login
from auth import admin  # import your admin router
# from fastapi import Request
from backend.Routes import conversations

from backend.b2_service import upload_file_to_b2, generate_download_url
from fastapi.responses import FileResponse, RedirectResponse



@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    ensure_auth_columns()
    ensure_conversation_columns()
    yield


app = FastAPI(title="AutoPrepAI Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signup.router, prefix="/auth")
app.include_router(login.router, prefix="/auth")
app.include_router(admin.router)
app.include_router(conversations.router)
# @app.middleware("http")
# async def enforce_auth(request: Request, call_next):
#     if request.url.path.startswith("/auth") or request.url.path.startswith("/health"):
#         return await call_next(request)
    
#     try:
#         user = await get_current_user(request)
#         request.state.user = user
#     except HTTPException:
#         raise HTTPException(status_code=401, detail="Not authenticated")
    
#     return await call_next(request)

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
    current_user: User = Depends(get_current_user),
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
        conversation = Conversation(title="New Chat", user_id=current_user.id)
        db.add(conversation)
        db.flush()

    if dataset is None:
        raise HTTPException(status_code=400, detail="Dataset is required")

    file_bytes = await dataset.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty")

    # Upload input dataset to B2
    input_key = f"inputs/{conversation.id}/{int(time.time())}_{dataset.filename}"
    try:
        upload_file_to_b2(
            file_bytes,
            key=input_key,
            content_type=dataset.content_type or "application/octet-stream",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset to storage: {exc}") from exc

    try:
        dataset_df = MLPipelineService.dataframe_from_upload(file_bytes, dataset.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {exc}") from exc

    # ── UPLOAD-ONLY GUARD ──────────────────────────────────────────────────────
    # When the frontend uploads a file it sends a descriptive sentence like
    # "I've uploaded a dataset: foo.csv …". This is not a preprocessing command,
    # so we return early without touching the NLP pipeline.
    clean_message = (message or "").strip()
    IS_UPLOAD_MESSAGE = (
        mode == "chat"  and (
        not clean_message
        or clean_message.lower().startswith("i've uploaded a dataset")
        or clean_message.lower().startswith("i have uploaded a dataset"))
    )

    if IS_UPLOAD_MESSAGE:
        upload_result = {
            "shape": list(dataset_df.shape),
            "logs": ["Dataset received and ready for processing."],
            "metadata": {},
            "data_preview": dataset_df.head(10).to_dict(orient="records"),
            "output_file": None,
            "download_url": None,
        }
        assistant_message = (
            f"Dataset loaded successfully! "
            f"I can see **{dataset_df.shape[0]} rows** and **{dataset_df.shape[1]} columns**. "
            f"Columns: {', '.join(dataset_df.columns.tolist())}. "
            f"What would you like me to do with it?"
        )
        safe_result = MLPipelineService._to_jsonable(upload_result)  # ← sanitize NaN → None

        db.add(ConversationMessage(
            conversation_id=conversation.id,
            role="user",
            content=clean_message or f"[uploaded: {dataset.filename}]",
            payload=None,
        ))
        db.add(ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=assistant_message,
            payload=json.loads(json.dumps(safe_result)),  # ← use sanitized version
        ))
        db.commit()
        return ChatResponse(
            conversation_id=conversation.id,
            assistant_message=assistant_message,
            result=safe_result,  
        )
    # ── END UPLOAD-ONLY GUARD ──────────────────────────────────────────────────

    try:
        result = MLPipelineService.process_message(
            user_message=clean_message,
            dataset_df=dataset_df,
            mode=mode,
            selected_intents=parsed_selected_intents,
            conversation_id=str(conversation.id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_key = result.get("output_file")
    if output_key:
        try:
            result["download_url"] = generate_download_url(output_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {exc}") from exc

    assistant_message = result.pop("assistant_message", "Processing completed successfully.")

    stored_message = clean_message or f"[{mode}]"

    db.add(ConversationMessage(
        conversation_id=conversation.id,
        role="user",
        content=stored_message,
        payload=None,
    ))
    db.add(ConversationMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_message,
        payload=json.loads(json.dumps(MLPipelineService._to_jsonable(result))),
    ))
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        assistant_message=assistant_message,
        result=result,
    )


@app.get("/download/{path:path}")
def download_processed_file(
    path: str,
    current_user: User = Depends(get_current_user)
):
    try:
        url = generate_download_url(path)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="File not found or could not generate download link") from exc

    return RedirectResponse(url=url)


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str,current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid conversation_id") from exc

    conversation = db.get(Conversation, conversation_uuid)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    #security check to ensure that the conversation belongs to the current user
    if conversation.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return conversation

@app.get("/conversations")
def get_user_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).all()

    return conversations
