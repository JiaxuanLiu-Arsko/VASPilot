from .server_listener import CrewServer
class MessageListener:
    def __init__(self, crew_server: CrewServer):
        self.user_messages = {}
        self.agent_messages = {}
        self.crew_server = crew_server

    def send_user_message(self, message: str, conversation_id: str):
        self.user_messages[conversation_id] = message
        self.crew_server.save_conversation_messages(conversation_id, 'User', message)

    def send_agent_message(self, message: str, conversation_id: str):
        self.agent_messages[conversation_id] = message
        self.crew_server.save_conversation_messages(conversation_id, "VASPilot", message)

    def read_user_message(self, conversation_id: str):
        message = self.user_messages.get(conversation_id, None)
        if message:
            del self.user_messages[conversation_id]
            return message
        else:
            return None