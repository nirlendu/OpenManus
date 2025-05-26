class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class SuperAgentError(Exception):
    """Base exception for all SuperAgent errors"""

    pass


class TokenLimitExceeded(SuperAgentError):
    """Exception raised when the token limit is exceeded"""

    pass
