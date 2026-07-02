"""
Pipeline: Orchestrates the execution of preprocessing agents.
"""
from typing import List
import logging
import copy

from business_logic.cleaning_coordinator.data_context import DataContext
from business_logic.cleaning_coordinator.pipeline_node import PipelineNode
from ml_layer.nlp.nlp_service import NLPService

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

    def run_single_agent(self, context: DataContext, session, user_command: str = "") -> tuple[DataContext, bool]:
        """
        Run the pipeline on the given context.

        Uses session["agent_index"] to track position so self.agents is never mutated.
        
        Args:
            context: Data context to process
            user_command: Optional user command for intent extraction
            
        Returns:
            Updated DataContext after all agents have run
        """
        if not self.agents:
            logging.getLogger(__name__).info("No agents to run")
            return context, True

        self.logger.info(f"Starting pipeline execution with {len(self.agents)} agents")

        # Make chat/manual command available to agents (notably NLPAgent).
        context.metadata["user_command"] = user_command or ""

        agent_index = session.get("agent_index", 0)
        execute = False

        while not execute:
            if agent_index >= len(self.agents):
                break
            node = self.agents[agent_index]
            context, execute = self._execute_node(node, context)
            session["last_executed_step"] = node.get_agent_name()
            agent_index += 1
            session["agent_index"] = agent_index

        self.logger.info("Pipeline execution completed")
        done = self.check_no_agents_left_to_run(context, session)
        print(f"[DEBUG] agent_index={agent_index}, total_agents={len(self.agents)}, done={done}")
        return context, done


    
    def check_no_agents_left_to_run(self,context: DataContext,session) -> bool:
        agent_index = session.get("agent_index", 0)
        while agent_index < len(self.agents):
            node = self.agents[agent_index]
            
            if node.should_run(context):
                print(f"[DEBUG] Node {node.get_agent_name()} should run")
                return False
            else:
                print(f"[DEBUG] Node {node.get_agent_name()} should NOT run")
            agent_index += 1
        return True
    #run automated
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
            context, _ = self._execute_node(node, context)
        
        # Save execution if session manager is available
        if self.session_manager and user_command:
            self._save_execution(user_command, context)
        
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

    def print_pipeline(self, context: DataContext) -> None:
        for node in self.agents:
            print(f"Agent: {node.get_agent_name()} | Should Run: {node.should_run(context)}")
    
    def _execute_node(self, node: PipelineNode, context: DataContext) -> tuple[DataContext, bool]:
        """
        Execute a single pipeline node.
        
        Args:
            node: The pipeline node to execute
            context: Current data context
            
        Returns:
            Updated DataContext and a boolean indicating if the node is done
        """
        agent_name = node.get_agent_name()
        # Check if node should run
        if not node.should_run(context):
            context.log(f"Skipping agent: {agent_name}")
            self.logger.info(f"Skipping agent: {agent_name}")
            return context,False
        
        # Snapshot data + metadata before execution
        data_before = context.data.copy()
        metadata_before = copy.deepcopy(context.metadata)
        
        # Execute the agent
        self.logger.info(f"Executing agent: {agent_name}")
        context.log(f"Executing agent: {agent_name}")
        
        try:
            context = node.execute(context)
        except Exception as e:
            error_msg = f"Error executing {agent_name}: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            return context, False
        
        # If data is unchanged, auto-advance without waiting for feedback
        if context.data.equals(data_before):
            context.log(f"No data changes detected — continuing to next step.")
            self.logger.info(f"No data changes for {agent_name}, auto-advancing")
            print(f"[DEBUG] No data changes for {agent_name}, auto-advancing")
            return context, False
        # Generate explanation if NLP service is available
        if self.nlp_service and agent_name != "NLP":
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
        
        return context,True

    def _save_execution(
        self, 
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
