import { useState, useRef, useCallback } from "react";
import {
  formatTime,
  updateDatasetFromResponse,
  userMsg,
  botMsg,
} from "../utils/helpers";
import { sendChatMessage, resetConversationData } from "../../../api/chat";
import { getAuthToken } from "../../../api/auth";
import { parseDatasetFile } from "../utils/fileParsers";
import { validateDatasetFile } from "../utils/helpers";
import {
  reconstructDatasetFile,
  resetDatasetState,
} from "../utils/datasetFile";
import { FileSpreadsheet } from "lucide-react";
import { appendUploadMessages, appendMessageToChats } from "../utils/chatState";
import { cleanError } from "../utils/handleErrors";

export function useDatasetState() {
  const [uploaded, setUploaded] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [rows, setRows] = useState(0);
  const [columns, setColumns] = useState(0);
  const [uploadError, setUploadError] = useState("");
  const [tableData, setTableData] = useState([]);
  const [tableDataBefore, setTableDataBefore] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [headersBefore, setHeadersBefore] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const [originalExtension, setOriginalExtension] = useState(".csv");
  const fileInputRef = useRef(null);
  const [isResetting, setIsResetting] = useState(false);

  const handleUploadClick = () => fileInputRef.current?.click();

  // Single place that knows how to write dataset state into all the setters
  const applyDatasetState = useCallback((state) => {
    setUploaded(state.uploaded);
    setDatasetName(state.datasetName);
    setRows(state.rows);
    setColumns(state.columns);
    setTableData(state.tableData);
    setTableDataBefore(state.tableDataBefore);
    setHeaders(state.headers);
    setHeadersBefore(state.headersBefore);
    setUploadError(state.uploadError);
    setOriginalExtension(state.originalExtension);
  }, []);

  const restoreDatasetFromPayload = useCallback(
    (payload) => {
      if (!payload?.dataset?.length) {
        if (payload?.dataset_name) {
          const ext =
            payload.dataset_name.match(/\.\w+$/)?.[0]?.toLowerCase() || ".csv";
          setDatasetName(payload.dataset_name);
          setOriginalExtension(ext);
        }
        return;
      }

      const headers = Object.keys(payload.dataset[0]);
      const beforeData = payload.data_preview_before ?? [];
      const beforeHeaders = beforeData.length
        ? Object.keys(beforeData[0])
        : headers;

      applyDatasetState({
        uploaded: true,
        datasetName: payload.dataset_name,
        rows: payload.shape?.[0] ?? payload.dataset.length,
        columns: payload.shape?.[1] ?? headers.length,
        tableData: payload.dataset,
        tableDataBefore: beforeData,
        headers,
        headersBefore: beforeHeaders,
        uploadError: "",
        originalExtension: payload.dataset_name
          ? payload.dataset_name.match(/\.\w+$/)?.[0]?.toLowerCase()
          : ".csv",
      });
    },
    [applyDatasetState],
  );

  const restoreUploadedDataset = useCallback(
    ({ fileName, rows, columns, data, headers, extension }) => {
      applyDatasetState({
        uploaded: true,
        datasetName: fileName,
        rows,
        columns,
        tableData: data,
        tableDataBefore: data,
        headers,
        headersBefore: headers,
        uploadError: "",
        originalExtension: extension,
      });
    },
    [applyDatasetState],
  );

  const handleReset = async (currentConversationId) => {
    if (isResetting) return;
    const confirmed = window.confirm(
      "Are you sure you want to reset all uploaded data? This action cannot be undone.",
    );
    if (!confirmed) return;

    setIsResetting(true);
    try {
      if (currentConversationId) {
        try {
          await resetConversationData(currentConversationId);
        } catch (err) {
          console.warn("Backend reset failed, cleaning frontend only:", err);
        }
      }

      resetDatasetState({
        setUploaded,
        setDatasetName,
        setRows,
        setColumns,
        setTableData,
        setTableDataBefore,
        setHeaders,
        setHeadersBefore,
        setUploadError,
        setOriginalExtension,
      });
    } finally {
      setIsResetting(false);
    }
  };

  const handleFileUpload = async (
    event,
    {
      activeChatId,
      currentConversationId,
      syncConversationId,
      setChats,
      setIsLoadingChat = () => {},
      setChatError = () => {},
      bumpChatToTop,
    } = {},
  ) => {
    const file = event.target.files?.[0];

    try {
      if (!file) return;

      if (!getAuthToken()) {
        throw new Error(
          "You must be logged in to upload files. Please log in first.",
        );
      }

      setIsLoadingChat(true);
      setChatError("");
      setUploadError("");

      // Validate
      const { isCSV, extension } = validateDatasetFile(file);

      // Parse locally
      const parsed = await parseDatasetFile(file, isCSV);

      const uploadMessage =
        `I've uploaded a dataset: ${file.name}. ` +
        `It has ${parsed.rows} rows and ` +
        `${parsed.columns} columns.`;

      // Send to backend
      const response = await sendChatMessage({
        message: uploadMessage,
        mode: "chat",
        selectedIntents: [],
        conversationId: currentConversationId,
        dataset: file,
      });

      const realConvId = response.conversation_id ?? currentConversationId;

      syncConversationId?.(response.conversation_id, activeChatId);

      // Restore frontend dataset
      restoreUploadedDataset({
        fileName: file.name,
        rows: parsed.rows,
        columns: parsed.columns,
        data: parsed.data,
        headers: parsed.headers,
        extension,
      });

      // Update chat
      appendUploadMessages({
        setChats,
        activeChatId,
        realConvId,
        uploadMessage,
        assistantMessage:
          response.assistant_message ?? `Dataset loaded: ${file.name}`,
        downloadUrl: response.result?.download_url,
      });
      bumpChatToTop?.(realConvId || activeChatId);
    } catch (error) {
      console.error("File upload error:", error);

      const message = cleanError(error.message);

      setUploadError(message);
      setChatError(message);
    } finally {
      setIsLoadingChat(false);
      event.target.value = "";
    }
  };

  const handleAutoClean = async (chatHelpers = {}) => {
    const {
      activeChatId,
      currentConversationId,
      syncConversationId,
      setChats,
      setIsLoadingChat,
      setChatError,
      setPendingFeedback,
      bumpChatToTop,
    } = chatHelpers;

    const safeSetChats = setChats ?? (() => {});
    const safeSetLoading = setIsLoadingChat ?? (() => {});
    const safeSetChatError = setChatError ?? (() => {});
    const safeSetPending = setPendingFeedback ?? (() => {});

    const AUTO_CLEAN_MESSAGE = "🤖 Auto-clean my dataset";

    if (!uploaded || !tableData?.length || !headers?.length) {
      safeSetChatError("Upload a dataset first");
      return;
    }

    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    safeSetChatError("");
    safeSetLoading(true);

    const time = formatTime();

    appendMessageToChats(
      safeSetChats,
      [thisChatId],
      userMsg(AUTO_CLEAN_MESSAGE, time),
    );

    try {
      const datasetFile = reconstructDatasetFile(
        headers,
        tableData,
        originalExtension,
        datasetName,
      );

      if (!datasetFile) {
        throw new Error("Could not reconstruct dataset.");
      }

      const response = await sendChatMessage({
        message: AUTO_CLEAN_MESSAGE,
        mode: "full_auto",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: datasetFile,
      });

      const { conversation_id, assistant_message, result, finished } = response;

      const realConvId = conversation_id ?? thisConvId;

      syncConversationId?.(conversation_id, thisChatId);

      safeSetPending(finished === false);

      updateDatasetFromResponse(result, {
        setHeaders,
        setHeadersBefore,
        setTableData,
        setTableDataBefore,
        setRows,
        setColumns,
      });

      appendMessageToChats(
        safeSetChats,
        [thisChatId, realConvId],
        botMsg(
          assistant_message ?? "✨ Auto-clean completed successfully.",
          formatTime(),
          result?.download_url,
        ),
      );
      bumpChatToTop?.(thisChatId);
    } catch (error) {
      console.error("Auto-clean error:", error);

      const friendlyMsg = cleanError(error.message);

      safeSetChatError(friendlyMsg);

      appendMessageToChats(
        safeSetChats,
        [thisChatId],
        botMsg(`❌ ${friendlyMsg}`, formatTime()),
      );
    } finally {
      safeSetLoading(false);
    }
  };

  const handleDownload = (messages) => {
    const last = [...messages].reverse().find((m) => m.downloadUrl);
    if (last?.downloadUrl) window.open(last.downloadUrl, "_blank");
  };

  const hasDownloadUrl = (messages) =>
    messages?.some((m) => m.downloadUrl) ?? false;

  const DatasetIcon = FileSpreadsheet;

  return {
    uploaded,
    setUploaded,
    datasetName,
    setDatasetName,
    rows,
    setRows,
    columns,
    setColumns,
    uploadError,
    setUploadError,
    tableData,
    setTableData,
    tableDataBefore,
    setTableDataBefore,
    headers,
    setHeaders,
    headersBefore,
    setHeadersBefore,
    showPreview,
    setShowPreview,
    originalExtension,
    setOriginalExtension,
    fileInputRef,
    handleUploadClick,
    handleReset,
    restoreDatasetFromPayload,
    handleFileUpload,
    handleAutoClean,
    handleDownload,
    hasDownloadUrl,
    DatasetIcon,
    isResetting,
    setIsResetting,
  };
}
