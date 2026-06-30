import { useState, useRef, useCallback } from "react";
import { cleanError, formatTime, updateDatasetFromResponse, userMsg, botMsg } from "../utils/helpers";
import { sendChatMessage, sendFeedback } from "../../../api/chat";
import { getAuthToken } from "../../../api/auth";
import { parseCSV, parseXLSX } from "../utils/fileParsers";
import { ALLOWED_MIME_TYPES } from "../constants";
import { reconstructDatasetFile } from "../utils/datasetFile";
import { FileSpreadsheet } from "lucide-react";


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

  const handleUploadClick = () => fileInputRef.current?.click();

  const restoreDatasetFromPayload = useCallback((payload) => {
    if (!payload) return;
    if (payload.dataset_name) {
      setDatasetName(payload.dataset_name);
      const ext = payload.dataset_name.match(/\.\w+$/)?.[0]?.toLowerCase() || ".csv";
      setOriginalExtension(ext);
    }
    if (payload.dataset?.length) {
      const newHeaders = Object.keys(payload.dataset[0]);
      setHeaders(newHeaders);
      const beforeData = payload.data_preview_before ?? [];
      const beforeHeaders = beforeData.length ? Object.keys(beforeData[0]) : newHeaders;
      setHeadersBefore(beforeHeaders);
      setTableData(payload.dataset);
      setTableDataBefore(beforeData);
      setRows(payload.shape?.[0] ?? payload.dataset.length);
      setColumns(payload.shape?.[1] ?? newHeaders.length);
      setUploaded(true);
    }
  }, []);

  const handleReset = () => {
    const confirmed = window.confirm("Are you sure you want to reset all uploaded data? This action cannot be undone.");
    if (!confirmed) return;
    setUploaded(false);
    setDatasetName("");
    setRows(0);
    setColumns(0);
    setTableData([]);
    setTableDataBefore([]);
    setHeaders([]);
    setHeadersBefore([]);
    setUploadError("");
    setOriginalExtension(".csv");
  };

  const handleResetWithBackend = async (currentConversationId) => {
    if (currentConversationId && tableDataBefore.length > 0) {
      try {
        await sendFeedback({ conversationId: currentConversationId, accept: false });
      } catch {
        // backend session may not exist — still reset frontend
      }
    }
    handleReset();
  };

  const handleFileUpload = async (event, chatHelpers = {}) => {
    const {
      activeChatId, currentConversationId,
      syncConversationId, setChats, setIsLoadingChat, setChatError,
    } = chatHelpers;

    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    if (!getAuthToken()) {
      setUploadError("You must be logged in to upload files. Please log in first.");
      event.target.value = "";
      return;
    }

    const fileName = selectedFile.name.toLowerCase();
    const isCSV = fileName.endsWith(".csv");
    const isXLSX = fileName.endsWith(".xlsx") || fileName.endsWith(".xls");

    if (!isCSV && !isXLSX) {
      setUploadError(`"${selectedFile.name}" is not supported. Please upload a CSV, or Excel file.`);
      event.target.value = "";
      return;
    }

    if (selectedFile.type && !ALLOWED_MIME_TYPES.includes(selectedFile.type)) {
      setUploadError(
        `"${selectedFile.name}" appears to be an image or unsupported file. Please upload a CSV, or Excel file.`
      );
      event.target.value = "";
      return;
    }

    setOriginalExtension(isXLSX ? (fileName.endsWith(".xls") ? ".xls" : ".xlsx") : ".csv");

    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;

    setIsLoadingChat?.(true);
    setChatError?.("");

    try {
      const parsed = isCSV
        ? parseCSV(await selectedFile.text())
        : await parseXLSX(selectedFile);

      setDatasetName(selectedFile.name);
      setRows(parsed.rows);
      setColumns(parsed.columns);
      setTableData(parsed.data);
      setHeaders(parsed.headers);
      setHeadersBefore(parsed.headers);
      setUploadError("");
      setUploaded(true);

      const response = await sendChatMessage({
        message: `I've uploaded a dataset: ${selectedFile.name}. It has ${parsed.rows} rows and ${parsed.columns} columns.`,
        mode: "chat",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: selectedFile,
      });

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId?.(response.conversation_id, thisChatId);

      const time = formatTime();
      const userUploadText = `I've uploaded a dataset: ${selectedFile.name}. It has ${parsed.rows} rows and ${parsed.columns} columns.`;

      setChats?.((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  userMsg(userUploadText, time),
                  botMsg(response.assistant_message || `✓ Dataset loaded: ${selectedFile.name}`, time, response.result?.download_url),
                ],
              }
            : chat
        )
      );
    } catch (error) {
      console.error("File upload error:", error);
      setChatError?.(cleanError(error.message));
      setUploadError(cleanError(error.message));
    } finally {
      setIsLoadingChat?.(false);
      event.target.value = "";
    }
  };

  const handleAutoClean = async (chatHelpers = {}) => {
    const {
      activeChatId, currentConversationId,
      syncConversationId, setChats, setIsLoadingChat, setChatError, setPendingFeedback,
    } = chatHelpers;

    if (!uploaded) {
      setChatError?.("Upload a dataset first");
      return;
    }

    setChatError?.("");
    setIsLoadingChat?.(true);

    const thisChatId = activeChatId;
    const thisConvId = currentConversationId;
    const time = formatTime();

    setChats?.((prev) =>
      prev.map((chat) =>
        chat.id === thisChatId
          ? {
              ...chat,
              messages: [
                ...chat.messages,
                userMsg("🤖 Auto-clean my dataset", time),
              ],
            }
          : chat
      )
    );

    try {
      const file = reconstructDatasetFile(headers, tableData, originalExtension, datasetName);

      const response = await sendChatMessage({
        message: "🤖 Auto-clean my dataset",
        mode: "full_auto",
        selectedIntents: [],
        conversationId: thisConvId,
        dataset: file,
      });

      setPendingFeedback?.(false);

      const realConvId = response.conversation_id ?? thisConvId;
      syncConversationId?.(response.conversation_id, thisChatId);

      updateDatasetFromResponse(response.result, { setHeaders, setHeadersBefore, setTableData, setTableDataBefore, setRows, setColumns });

      const botTime = formatTime();

      setChats?.((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId || chat.id === realConvId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  botMsg(response.assistant_message || "✨ Auto-clean completed successfully.", botTime, response.result?.download_url),
                ],
              }
            : chat
        )
      );
    } catch (error) {
      console.error(error);
      const friendlyMsg = cleanError(error.message);
      setChatError?.(friendlyMsg);
      const botTime = formatTime();
      setChats?.((prev) =>
        prev.map((chat) =>
          chat.id === thisChatId
            ? {
                ...chat,
                messages: [
                  ...chat.messages,
                  botMsg(`❌ ${friendlyMsg}`, botTime),
                ],
              }
            : chat
        )
      );
    } finally {
      setIsLoadingChat?.(false);
    }
  };

  const handleDownload = (messages) => {
    const last = [...messages].reverse().find((m) => m.downloadUrl);
    if (last) window.open(last.downloadUrl, "_blank");
  };

  const hasDownloadUrl = (messages) => messages?.some((m) => m.downloadUrl) ?? false;

  const csvFileName = datasetName
    ? datasetName.replace(/\.\w+$/, ".csv")
    : "data.csv";

  const DatasetIcon = FileSpreadsheet;

  return {
    uploaded, setUploaded,
    datasetName, setDatasetName,
    rows, setRows,
    columns, setColumns,
    uploadError, setUploadError,
    tableData, setTableData,
    tableDataBefore, setTableDataBefore,
    headers, setHeaders,
    headersBefore, setHeadersBefore,
    showPreview, setShowPreview,
    originalExtension, setOriginalExtension,
    fileInputRef,
    handleUploadClick,
    handleReset,
    handleResetWithBackend,
    restoreDatasetFromPayload,
    handleFileUpload,
    handleAutoClean,
    handleDownload,
    hasDownloadUrl,
    csvFileName,
    DatasetIcon,
  };
}
