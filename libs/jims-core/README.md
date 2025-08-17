# Name

JIMS stands for "Just an Integrated Multiagent System"

# Concepts

## Thread

Thread is a conversation between user (or users) and agentic system
Each interaction between agents and users happens in thread

## Event

Something that happens in a thread
Each event has id, type and data

Example:

{
    "event_id": "...",
    "event_type": "comm.user_message",
    "event_data": {
        "content": "Hello!"
    }
}
