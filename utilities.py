# agents = []
# dataset_before = None
# dataset_after = None
# previous_logs = None
# finished = None
# metadata = None
# mode = None
# #add the global variable in here in the main inside backend/main.py and pipeline.py
sessions = {}
# Each entry: sessions[conversation_id] = {
#     "pipeline": ...,
#     "dataset_before": ...,
#     "dataset_after": ...,
#     "previous_logs": [],
#     "context_metadata": {},
#     "finished": False,
#     "mode": "",
#     "agents": []
#     "context": None
# }