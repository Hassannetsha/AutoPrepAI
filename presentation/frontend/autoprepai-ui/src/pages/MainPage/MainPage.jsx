import { useState } from "react";
import "../../styles/style.css";
import ActionList from "../../components/main/ActionList";
import AppHeader from "../../components/main/AppHeader";
import ChatInput from "../../components/main/ChatInput";
import ChatSidebar from "../../components/main/ChatSidebar";
import ChatWindow from "../../components/main/ChatWindow";
import DataPreviewModal from "../../components/main/DataPreviewModal";
import DatasetSidebar from "../../components/main/DatasetSidebar";
import HistoryModal from "../../components/main/HistoryModal";
import { ACTIONS } from "./constants";
import { useChatManager } from "./hooks/useChatManager";
import { useDatasetState } from "./hooks/useDatasetState";
import { useHistory } from "./hooks/useHistory";

export default function MainPage() {
  const [selectedActions, setSelectedActions] = useState([]);
  const [inputValue, setInputValue] = useState("");

  const dataSet = useDatasetState();

  const chat = useChatManager({
    setUploaded: dataSet.setUploaded,
    setTableData: dataSet.setTableData,
    setTableDataBefore: dataSet.setTableDataBefore,
    setHeaders: dataSet.setHeaders,
    setHeadersBefore: dataSet.setHeadersBefore,
    setDatasetName: dataSet.setDatasetName,
    setRows: dataSet.setRows,
    setColumns: dataSet.setColumns,
    setOriginalExtension: dataSet.setOriginalExtension,
    setUploadError: dataSet.setUploadError,
    restoreDatasetFromPayload: dataSet.restoreDatasetFromPayload,
    setSelectedActions,
  });

  const history = useHistory({
    currentConversationId: chat.currentConversationId,
    setChatError: chat.setChatError,
  });

  const toggleAction = (action) => {
    setSelectedActions((prev) =>
      prev.includes(action) ? prev.filter((a) => a !== action) : [...prev, action]
    );
  };

  const canSend = dataSet.uploaded && (inputValue.trim() !== "" || selectedActions.length > 0);

  const chatHelpers = {
    activeChatId: chat.activeChatId,
    currentConversationId: chat.currentConversationId,
    syncConversationId: chat.syncConversationId,
    setChats: chat.setChats,
    setIsLoadingChat: chat.setIsLoadingChat,
    setChatError: chat.setChatError,
    setPendingFeedback: chat.setPendingFeedback,
    bumpChatToTop: chat.bumpChatToTop,
  };

  return (
    <div className="container">
      <ChatSidebar
        sidebarCollapsed={chat.sidebarCollapsed}
        setSidebarCollapsed={chat.setSidebarCollapsed}
        handleNewChat={chat.handleNewChat}
        chats={chat.chats}
        activeChatId={chat.activeChatId}
        setActiveChatId={chat.handleSwitchChat}
        handleRenameChat={chat.handleRenameChat}
        handleDeleteChat={chat.handleDeleteChat}
      />

      <DatasetSidebar
        fileInputRef={dataSet.fileInputRef}
        handleFileUpload={(e) => dataSet.handleFileUpload(e, chatHelpers)}
        uploaded={dataSet.uploaded}
        handleUploadClick={dataSet.handleUploadClick}
        uploadError={dataSet.uploadError}
        DatasetIcon={dataSet.DatasetIcon}
        datasetName={dataSet.datasetName}
        rows={dataSet.rows}
        columns={dataSet.columns}
        setShowPreview={dataSet.setShowPreview}
        handleDownload={() => dataSet.handleDownload(chat.activeChat?.messages)}
        hasDownloadUrl={dataSet.hasDownloadUrl(chat.activeChat?.messages)}
        handleShowHistory={history.handleShowHistory}
        handleReset={() => dataSet.handleReset(chat.currentConversationId)}
        isResetting={dataSet.isResetting}
        handleAutoClean={() => dataSet.handleAutoClean(chatHelpers)}
        autoCleanDisabled={!dataSet.uploaded || chat.isLoadingChat || chat.pendingFeedback}
        pendingFeedback={chat.pendingFeedback}
      />

      <div className="main">
        <AppHeader />
        <ChatWindow
          activeChat={chat.activeChat}
          chatEndRef={chat.chatEndRef}
          pendingFeedback={chat.pendingFeedback}
          onAcceptFeedback={() => chat.handleFeedback(true)}
          onRejectFeedback={() => chat.handleFeedback(false)}
          feedbackDisabled={chat.submittingFeedback}
          stepTitle={chat.stepTitle}
        />
        <ActionList
          actions={ACTIONS}
          uploaded={dataSet.uploaded}
          selectedActions={selectedActions}
          toggleAction={toggleAction}
          onActionClick={toggleAction}
          isLoading={chat.isLoadingChat || chat.pendingFeedback}
        />
        {chat.chatError && (
          <div className="chat-error">{chat.chatError}</div>
        )}
        {chat.isLoadingChat && (
          <div className="chat-loading">Processing...</div>
        )}
        <ChatInput
          inputValue={inputValue}
          setInputValue={setInputValue}
          handleSend={() => chat.handleSend({
            inputValue,
            selectedActions,
            setInputValue,
            headers: dataSet.headers,
            tableData: dataSet.tableData,
            originalExtension: dataSet.originalExtension,
            datasetName: dataSet.datasetName,
          })}
          canSend={canSend}
          isLoading={chat.isLoadingChat || chat.pendingFeedback}
        />
      </div>

      {dataSet.showPreview && (
        <DataPreviewModal
          onClose={() => dataSet.setShowPreview(false)}
          data={dataSet.tableData}
          dataBefore={dataSet.tableDataBefore}
          headers={dataSet.headers}
          headersBefore={dataSet.headersBefore}
          datasetName={dataSet.datasetName}
        />
      )}
      {history.showHistory && (
        <HistoryModal
          onClose={() => history.setShowHistory(false)}
          logs={history.historyLogs}
          loadingHistory={history.loadingHistory}
        />
      )}

      {(chat.isDeletingChat || dataSet.isResetting) && (
        <div className="page-loading-overlay">
          <div className="page-loading-spinner"></div>
        </div>
      )}
    </div>
  );
}