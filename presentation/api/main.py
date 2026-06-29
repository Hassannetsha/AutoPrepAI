import copy
import json
import uuid
import time
from pathlib import Path

import pandas as pd

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from business_logic.auth.dependencies import get_current_user
from data_access.database.connection import Base, engine, ensure_auth_columns, ensure_conversation_columns, get_db
from business_logic.cleaning_coordinator.ml_service import MLPipelineService
from data_access.database.models import Conversation, ConversationMessage, User, WELCOME_TEXT
from presentation.api.schemas import ChatResponse, ConversationOut, FeedbackRequest
from business_logic.auth import signup, login
from business_logic.auth import admin  # import your admin router
# from fastapi import Request
from presentation.api.routes import conversations

from data_access.storage.b2_service import upload_file_to_b2, generate_download_url
from fastapi.responses import FileResponse, RedirectResponse
from business_logic.services import session_store as utilities


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
    print(f"Received /chat request with conversation_id={conversation_id}, mode={mode}, selected_intents={selected_intents}")
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
        db.add(ConversationMessage(
            conversation_id=conversation.id,
            role="assistant",
            content=WELCOME_TEXT,
            payload=None,
        ))
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

    # ── SIZE LIMITS (before B2 upload to avoid wasting bandwidth) ──────────
    # Allowed: 50000 rows x 100 columns, or 100 rows x 50000 columns, or smaller
    n_rows, n_cols = dataset_df.shape
    if not ((n_rows <= 50000 and n_cols <= 100) or (n_rows <= 100 and n_cols <= 50000)):
        raise HTTPException(
            status_code=413,
            detail=f"Dataset too large: {n_rows} rows x {n_cols} columns. "
                   f"Maximum allowed is 50000 rows x 100 columns "
                   f"(or 100 rows x 50000 columns in the reverse case)."
        )
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
            "dataset_name": dataset.filename,
            "shape": list(dataset_df.shape),
            "logs": ["Dataset received and ready for processing."],
            "metadata": {},
            # "data_preview": dataset_df.head(10).to_dict(orient="records"),
            "dataset": dataset_df.to_dict(orient="records"),
            "data_preview_before": dataset_df.to_dict(orient="records"),
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
    mode = MLPipelineService._normalize_mode(mode)
    print(f"{conversation.id}: Starting processing with mode={mode}, selected_intents={parsed_selected_intents}")
    # utilities.sessions.setdefault(str(conversation.id), MLPipelineService.session_builder(conversation.id))["dataset_before"] = dataset_df.copy()
    orig_ext = Path(dataset.filename).suffix if dataset.filename else ".csv"
    orig_name = dataset.filename or f"data{orig_ext}"
    existing_session = utilities.sessions.get(str(conversation.id))
    if existing_session is None or existing_session.get("finished", False):
        utilities.sessions[str(conversation.id)] = MLPipelineService.session_builder(
            str(conversation.id), mode, file_extension=orig_ext, original_filename=orig_name
        )
    utilities.sessions[str(conversation.id)]["dataset_before"] = dataset_df.copy()
    #this will route to the new endpoint for manual and chat modes where the frontend will handle the step by step execution and user feedback
    # and take the part of manual in the function process message as there will be no user input
    session = utilities.sessions.get(str(conversation.id))
    print(f"Session at start of /chat: {session}")
    # if session is None:
    #         # First call for this conversation — build a fresh pipeline
    #         session = MLPipelineService.session_builder(conversation.id)
    try:
        result,session["finished"] = MLPipelineService.process_message(
            user_message=clean_message,
            dataset_df=dataset_df,
            mode=mode,
            selected_intents=parsed_selected_intents,
            conversation_id=str(conversation.id),
        )
        # if mode=="manual" or mode == "chat":
        #     session["finished"] = False
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    output_key = result.get("output_file")
    if output_key:
        try:
            result["download_url"] = generate_download_url(output_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {exc}") from exc
    # print(f"[DEBUG] Session after processing in line 209")
    assistant_message = result.pop("assistant_message", "Processing completed successfully.")
    # print(f"[DEBUG] Session after processing in line 211")
    finished = True
    if session["dataset_before"].equals(session["dataset_after"]):
        assistant_message += "\n[System] No changes were made to the dataset in this step."
        if MLPipelineService.check_no_agents_left_to_run(session["dataset_before"], session, session["pipeline"]):
            finished = True
            utilities.sessions.pop(conversation_id, None)
        print(f"[DEBUG] In line 320")
    elif mode=="manual" or mode == "chat" and "sorry your request" not in assistant_message.lower():
        assistant_message += "\n[System] Waiting for your feedback to proceed to the next step."
        finished = False
    stored_message = clean_message or f"[{mode}]"
    # if :
    #     finished = True
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
        payload={
            **json.loads(json.dumps(MLPipelineService._to_jsonable(result))),
            "finished": finished,
            "step_title": session.get("last_executed_step") or "",
            "agent_index": session.get("agent_index", 0),
            "mode": mode,
            "file_extension": orig_ext,
            "original_filename": orig_name,
        },
    ))
    db.commit()

    safe_result = MLPipelineService._to_jsonable(result)  # ← sanitize NaN → None

    return ChatResponse(
        conversation_id=conversation.id,
        assistant_message=assistant_message,
        result=safe_result,
        finished=finished,
        step_title=session.get("last_executed_step") or "",
    )
    # else:
    #     raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

@app.post("/chat/feedback", response_model=ChatResponse)
async def chat_feedback(
    body: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        conversation_uuid = uuid.UUID(body.conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation_id")
    session = utilities.sessions.get(str(conversation_uuid))
    if session is None:
        # Rebuild session from the last assistant message payload (survives server restarts)
        conversation = db.get(Conversation, conversation_uuid)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        last_msg = None
        for msg in reversed(conversation.messages or []):
            if msg.role in ("assistant", "bot") and msg.payload:
                last_msg = msg
                break
        if not last_msg or not last_msg.payload:
            raise HTTPException(status_code=400, detail="No active session for this conversation. Please send a message to start a new session.")
        pl = last_msg.payload
        session = MLPipelineService.session_builder(
            str(conversation_uuid),
            pl.get("mode", "chat"),
            file_extension=pl.get("file_extension", ".csv"),
            original_filename=pl.get("original_filename", "data.csv"),
        )
        session["agent_index"] = pl.get("agent_index", 0)
        session["last_executed_step"] = pl.get("step_title", "")
        session["previous_logs"] = pl.get("logs", [])
        session["finished"] = pl.get("finished", True)
        session["context_metadata"] = pl.get("metadata", {})
        # Rebuild datasets from JSON records
        dataset_records = pl.get("dataset") or []
        before_records = pl.get("data_preview_before") or dataset_records
        session["dataset_after"] = pd.DataFrame(dataset_records) if dataset_records else pd.DataFrame()
        session["dataset_before"] = pd.DataFrame(before_records) if before_records else pd.DataFrame()
        # Preserve result fields, excluding session-only keys
        session_keys = {"finished", "step_title", "agent_index", "mode", "file_extension", "original_filename"}
        session["result"] = {k: v for k, v in pl.items() if k not in session_keys}
        utilities.sessions[str(conversation_uuid)] = session
    step_executed = session.get("last_executed_step", "last step")
    for step in session["pipeline"].agents:
        print(f"Previous log: {step.get_agent_name()}")
    if body.accept:
        dataset_df = session["dataset_after"].copy()
        # print(f"dataset_df before reverting: {dataset_df.head()}")
        session["previous_logs"].append(f"[System] User ACCEPTED changes from: {step_executed}")
    else:
        dataset_df = session["dataset_before"].copy()
        # print(f"dataset_df before reverting: {dataset_df.head()}")
        session["previous_logs"].append(f"[System] User REJECTED changes from: {step_executed}. Reverting data.")


    # 3. Check if we are out of steps
    finished = session.get("finished")
    print(f"[DEBUG] Finished status: {finished}, dataset shape: {dataset_df.shape}, previous logs: {session['previous_logs']}")
    if not finished:
        try:
            result, session["finished"] = MLPipelineService.process_message(
                user_message="",
                dataset_df=dataset_df,
                mode=session["mode"],
                conversation_id=str(conversation_uuid),
            )
            # finished = session["finished"]
        except Exception as exc:
            print(f"[ERROR] {exc}")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        
        result = session["result"]
        if not body.accept:
            # result["data_preview"] = json.loads(
            #     dataset_df.head(50).to_json(orient="records")
            # )
            result["dataset"] = json.loads(
                dataset_df.to_json(orient="records")
            )
            result["data_preview_before"] = json.loads(
                dataset_df.to_json(orient="records")
            )
            result["shape"] = (int(dataset_df.shape[0]), int(dataset_df.shape[1]))
        result["logs"] = session["previous_logs"].copy()
        result["output_file"] = MLPipelineService.save_processed_dataframe(
            dataset_df, str(conversation_uuid),
            file_extension=session.get("file_extension", ".csv"),
            original_filename=session.get("original_filename")
        )
        utilities.sessions.pop(str(conversation_uuid), None)
        # session = utilities.sessions.get(str(conversation_uuid))
        # print(f"Session after pop: {utilities.sessions.get(str(conversation_uuid))}")
    output_key = result.get("output_file")
    if output_key:
        try:
            result["download_url"] = generate_download_url(output_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate download URL: {exc}") from exc
    print(f"[DEBUG] In line 316")
    assistant_message = result.pop("assistant_message", "Processing completed successfully.")
    if session["dataset_before"].equals(session["dataset_after"]):
        print(f"[DEBUG] In line 358")
        assistant_message += "\n[System] No changes were made to the dataset in this step."
        if MLPipelineService.check_no_agents_left_to_run(session["dataset_before"], session, session["pipeline"]):
            print(f"[DEBUG] In line 361")
            finished = True
            utilities.sessions.pop(str(conversation_uuid), None)
        print(f"[DEBUG] In line 320")
    if not finished:
        assistant_message += "\n[System] Waiting for your feedback to proceed to the next step."
        print(f"[DEBUG] In line 324")
    db.add(ConversationMessage(
        conversation_id=conversation_uuid,
        role="assistant",
        content=assistant_message,
        payload={
            **json.loads(json.dumps(MLPipelineService._to_jsonable(result))),
            "finished": finished,
            "step_title": session.get("last_executed_step") or "",
            "agent_index": session.get("agent_index", 0),
            "mode": session.get("mode", "chat"),
            "file_extension": session.get("file_extension", ".csv"),
            "original_filename": session.get("original_filename", "data.csv"),
        },
    ))
    db.commit()

    safe_result = MLPipelineService._to_jsonable(result) 

    return ChatResponse(
        conversation_id=conversation_uuid,
        assistant_message=assistant_message,
        result=safe_result,
        finished=finished,
        step_title=session.get("last_executed_step") or "",
    )
        

#     # 4. If more steps exist, download the approved data from B2
#     try:
#         file_bytes = download_file_from_b2(target_file_key)
#         dataset_df = MLPipelineService.dataframe_from_upload(file_bytes, filename="temp.csv")
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Failed to load dataset for next step: {exc}")

#     # 5. Pop the next intent and process it
#     next_intent = pending_intents.pop(0)
    
#     metadata_state = payload.get("metadata", {
#         "has_text": True, "has_numeric": True, "has_categorical": True, "nlp_done": True
#     })
#     metadata_state = {**metadata_state, "intents": [(next_intent, [])]}
    
#     result = MLPipelineService.process_single_step(
#         dataset_df=dataset_df,
#         intent=next_intent,
#         conversation_id=str(conversation_uuid),
#         metadata_state=metadata_state
#     )
    
#     # --- NEW: Combine the accumulated history with the logs of the new step ---
#     result["logs"] = previous_logs + result.get("logs", [])
#     result["pending_intents"] = pending_intents
#     result["output_file"] = result["proposed_data_key"]
    
#     try:
#         result["download_url"] = generate_download_url(result["proposed_data_key"])
#     except Exception:
#         pass

#     # 6. Save the user's choice and the next proposed step to the database
#     user_msg = "Accepted previous step." if body.accept else "Rejected previous step."
#     db.add(ConversationMessage(
#         conversation_id=conversation_uuid, role="user", content=user_msg, payload=None
#     ))
#     db.add(ConversationMessage(
#         conversation_id=conversation_uuid, 
#         role="assistant", 
#         content=f"Processed step: {next_intent}. Ready for review.", 
#         payload=result
#     ))
#     db.commit()

#     return ChatResponse(
#         conversation_id=conversation_uuid,
#         assistant_message=f"Processed step: {next_intent}. Ready for review.",
#         result=result,
# )
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
