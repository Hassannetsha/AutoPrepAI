"""
Pipeline: Orchestrates the execution of preprocessing agents.
"""
from typing import List
import logging
import copy

from data_context import DataContext
from pipeline_node import PipelineNode
from services.nlp_service import NLPService


class Pipeline:
    """
    Main pipeline orchestrator that executes preprocessing agents in sequence.
    """
    
    def __init__(
        self,
        agents: List[PipelineNode],
        session_manager=None,
        data_loader=None
    ):
        """
        Args:
            agents: List of pipeline nodes to execute
            session_manager: Optional session manager for persistence
            data_loader: Optional data loader for loading datasets
        """
        self.agents = agents
        self.session_manager = session_manager
        self.data_loader = data_loader
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # NLP service will be injected when needed
        self.nlp_service = NLPService()

    def set_nlp_service(self, nlp_service: NLPService):
        """Set the NLP service for intent extraction and explanations."""
        self.nlp_service = nlp_service

    def run(self, context: DataContext, user_command: str = "") -> DataContext:
        """
        Run the pipeline on the given context.
        
        Args:
            context: Data context to process
            user_command: Optional user command for intent extraction
            
        Returns:
            Updated DataContext after all agents have run
        """
        self.logger.info(f"Starting pipeline execution with {len(self.agents)} agents")

        # Make chat/manual command available to agents (notably NLPAgent).
        context.metadata["user_command"] = user_command or ""
        
        for node in self.agents:
            try:
                context = self._execute_node(node, context)
            except Exception as e:
                error_msg = f"Error executing {node.get_agent_name()}: {e}"
                self.logger.error(error_msg)
                context.log(error_msg)
                # Continue with next agent instead of failing completely
                continue
        
        # Save execution if session manager is available
        if self.session_manager and user_command:
            self._save_execution(context, user_command, context)
        
        self.logger.info("Pipeline execution completed")
        return context

    def add_agent(self, node: PipelineNode) -> None:
        """Add a new agent node to the pipeline."""
        self.agents.append(node)
        self.logger.info(f"Added agent: {node.get_agent_name()}")

    def remove_agent(self, agent_name: str) -> None:
        """Remove an agent from the pipeline by name."""
        self.agents = [n for n in self.agents if n.get_agent_name() != agent_name]
        self.logger.info(f"Removed agent: {agent_name}")

    def _execute_node(self, node: PipelineNode, context: DataContext) -> DataContext:
        """
        Execute a single pipeline node.
        
        Args:
            node: The pipeline node to execute
            context: Current data context
            
        Returns:
            Updated DataContext
        """
        agent_name = node.get_agent_name()
        
        # Check if node should run
        if not node.should_run(context):
            context.log(f"Skipping agent: {agent_name}")
            self.logger.info(f"Skipping agent: {agent_name}")
            return context
        
        # Snapshot metadata before execution
        metadata_before = copy.deepcopy(context.metadata)
        
        # Execute the agent
        self.logger.info(f"Executing agent: {agent_name}")
        context.log(f"Executing agent: {agent_name}")
        
        context = node.execute(context)
        
        # Generate explanation if NLP service is available
        if self.nlp_service and agent_name != "NLP":
            try:
                explanation = self.nlp_service.explain_step_llm(
                    step_name=agent_name,
                    metadata_before=metadata_before,
                    metadata_after=context.metadata
                )

                context.metadata.setdefault("explanations", []).append({
                    "step": agent_name,
                    "explanation": explanation
                })
                context.log(f"Explanation for '{agent_name}': {explanation}")
                
            except Exception as e:
                self.logger.warning(f"Failed to generate explanation for {agent_name}: {e}")
        
        return context

    def _save_execution(
        self, 
        context: DataContext, 
        user_command: str, 
        result: DataContext
    ) -> None:
        """Save execution details using session manager."""
        if not self.session_manager:
            return
        
        try:
            # Extract execution summary
            summary = {
                "command": user_command,
                "logs": result.logs,
                "metadata": result.metadata
            }
            
            # Save to session manager
            # (Actual implementation would depend on SessionManager interface)
            self.logger.info("Execution saved to session")
            
        except Exception as e:
            self.logger.error(f"Failed to save execution: {e}")
