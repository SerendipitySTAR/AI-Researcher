import time
# Potentially import LogAggregator and RealTimeMonitor if they were defined elsewhere,
# but for this step, they are conceptual as per the issue.

class UnifiedLogger:
    def __init__(self, main_logger=None, log_path_prefix="unified_log"):
        """
        Initializes the UnifiedLogger.
        main_logger: An optional existing logger instance (like MetaChainLogger) to also send logs to.
        log_path_prefix: Prefix for any files this logger might create (conceptual for now).
        """
        self.main_logger = main_logger
        # self.log_aggregator = LogAggregator() # Conceptual
        # self.real_time_monitor = RealTimeMonitor() # Conceptual
        
        # Simple internal log storage for demonstration
        self.log_records = []

        if self.main_logger:
            self.main_logger.info("[UnifiedLogger] Initialized.")
        else:
            print("[UnifiedLogger] Initialized.")

    def log_agent_activity(self, agent_name: str, activity_type: str, metadata: dict = None, message: str = ""):
        """
        Logs activity from an agent.
        agent_name: Name of the agent.
        activity_type: Type of activity (e.g., "start_call", "end_call", "error", "info").
        metadata: A dictionary of relevant data (e.g., input arguments, output summary).
        message: A specific message for the log entry.
        """
        timestamp = time.time()
        log_entry = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            "activity_type": activity_type,
            "message": message,
            "metadata": metadata or {}
        }
        self.log_records.append(log_entry)

        formatted_log = f"Agent: {agent_name}, Type: {activity_type}"
        if message:
            formatted_log += f", Msg: {message}"
        if metadata:
            # Avoid overly verbose metadata in the main log line, could summarize or select keys
            meta_summary = str(metadata)[:100] + "..." if len(str(metadata)) > 100 else str(metadata)
            formatted_log += f", Meta: {meta_summary}"

        if self.main_logger:
            self.main_logger.info(f"[UnifiedLogger] {formatted_log}")
        else:
            print(f"[UnifiedLogger] {formatted_log}")
        
        # Conceptual: Send to aggregator and monitor
        # self.log_aggregator.add_log(log_entry)
        # self.real_time_monitor.update_status(agent_name, activity_type, metadata)

    def get_all_records(self) -> list:
        return self.log_records

    def get_records_by_agent(self, agent_name: str) -> list:
        return [rec for rec in self.log_records if rec["agent_name"] == agent_name]
