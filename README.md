# AI Detection System - Modular Architecture

## Overview
Modular detection system with separated concerns for scalability and maintainability.

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   predict.py    │         │    agent.py     │
│   (Detection)   │────────▶│  (Alerts)       │
│                 │  Queue  │                 │
└─────────────────┘         └─────────────────┘
        ▲                            
        │                            
   ┌────┴────┐                      
   │launcher │                      
   │  .py    │                      
   └─────────┘                      
```

### Components

**1. predict.py** - Detection Process
- Runs YOLO model on video frames
- Detects objects and boundary crossings
- Sends events to multiprocessing queue
- Handles video visualization (GUI)
- Can run standalone (no alerts)

**2. agent.py** - Alert Agent
- Monitors event queue
- Generates GPT-based alert messages
- Sends email notifications
- Runs as separate process
- Cannot run standalone

**3. launcher.py** - System Coordinator
- Starts both processes
- Manages inter-process communication
- Handles graceful shutdown
- Main entry point

## Usage

### Full System (Detection + Alerts)
```bash
python launcher.py
```

### Detection Only (No Alerts)
```bash
python predict.py
```

## Configuration

Set in `.env` file:
```env
# Email Configuration
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
RECIPIENT_EMAIL=recipient@example.com

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
```

## Event Flow

1. **Detection** detects object → creates event dict
2. **Event** put in multiprocessing queue
3. **Agent** receives event from queue
4. **Agent** generates GPT message
5. **Agent** sends email alert

## Event Structure

### Detection Event
```python
{
    'type': 'detection',
    'class_name': 'fire',
    'count': 3,
    'timestamp': '2024-02-16 14:30:45',
    'frame': 1234
}
```

### Crossing Event
```python
{
    'type': 'crossing',
    'class_name': 'person',
    'count': 1,
    'direction': 'entering',
    'timestamp': '2024-02-16 14:31:12',
    'frame': 1567
}
```

## Event Limits

- **Max 2 events per video**:
  - 1 detection alert (first detection)
  - 1 crossing alert (first boundary crossing)

## Scaling

### Horizontal Scaling
```python
# Multiple detection processes
event_queue = mp.Queue()
detection1 = mp.Process(target=run_detection, args=(event_queue,))
detection2 = mp.Process(target=run_detection, args=(event_queue,))

# Single agent handles all
agent = mp.Process(target=run_agent, args=(event_queue,))
```

### Vertical Scaling
```python
# Multiple agents for different alert types
email_queue = mp.Queue()
sms_queue = mp.Queue()
webhook_queue = mp.Queue()

# Route events to appropriate queues
```

## Benefits

1. **Separation of Concerns**: Detection and alerting are independent
2. **Scalability**: Easy to add more detection processes or agents
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Changes to alerts don't affect detection
5. **Flexibility**: Can run detection without alerts

## Requirements

- Python 3.8+
- ultralytics (YOLO)
- opencv-python
- torch
- python-dotenv
- openai