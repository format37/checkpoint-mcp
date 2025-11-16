# checkpoint-mcp
Knowledge evaluation MCP server with visualization and analytics

## Overview
An MCP server designed to track and visualize learning progress across multiple topics through knowledge evaluations. The server provides both data tracking and intelligent analysis to help identify knowledge gaps and optimize learning paths.

### Key Features
- **Knowledge Evaluation**: Track learning progress with width (breadth) and depth scores
- **Progress Visualization**: Generate charts showing learning trends over time
- **Gap Analysis**: Identify knowledge gaps and receive personalized recommendations
- **Voice-Friendly**: Optimized for voice interactions with concise function names

## Core Concepts

### Evaluation Dimensions
Each topic is evaluated across two dimensions:
- **Width (Breadth)**: Ability to produce multiple valid solutions/approaches to a problem
- **Depth**: Ability to explain details, implementation, and theory of one approach

Scores range from 0.0 to 1.0 for both dimensions.

### Example Evaluation
```json
{
  "topic": "gradient descent",
  "answer": "Gradient descent minimizes loss by iteratively...",
  "width_score": 0.6,
  "width_explanation": "Covered SGD, momentum, and Adam",
  "depth_score": 0.7,
  "depth_explanation": "Detailed math and convergence discussion"
}
```

## Available Tools

### 1. `get_progress()`
Get overall learning progress for all topics in a single call.

**Returns**: JSON array with latest scores, trends, and evaluation counts for each topic.

**Use case**: Quick dashboard view of all learning progress.

### 2. `get_history(topic)`
Get chronological evaluation history for a single topic.

**Parameters**:
- `topic`: Topic name (e.g., "gradient descent")

**Returns**: Chronological array of all evaluations including scores and explanations.

**Use case**: Review how understanding has evolved over time.

### 3. `save_evaluation(topic, answer, width_score, width_explanation, depth_score, depth_explanation)`
Save new evaluation and compare with previous evaluation.

**Parameters**:
- `topic`: Topic being evaluated
- `answer`: User's answer text
- `width_score`: Breadth score (0.0-1.0)
- `width_explanation`: Explanation for width evaluation
- `depth_score`: Depth score (0.0-1.0)
- `depth_explanation`: Explanation for depth evaluation

**Returns**: JSON with score differences and previous evaluation data nested under "previous" key.

**Use case**: Record evaluation results and see improvement from last time.

### 4. `plot_topic(topic)` ✨ NEW
Generate line chart showing width and depth scores over time.

**Parameters**:
- `topic`: Topic to visualize

**Returns**: Image + statistics including trends and evaluation count.

**Use case**: Visual representation of learning progress for a specific topic.

### 5. `plot_all()` ✨ NEW
Generate summary visualization comparing all evaluated topics.

**Returns**: Two-subplot image:
- Horizontal bar chart comparing topics
- Scatter plot showing width vs depth relationship

**Use case**: Comprehensive view of learning progress across all topics.

### 6. `analyze_gaps(topics)` ✨ NEW
Analyze evaluation explanations to identify knowledge gaps and provide recommendations.

**Parameters**:
- `topics`: Optional list of topics to analyze (defaults to all)

**Returns**: Detailed analysis including:
- Knowledge gaps sorted by severity
- Common patterns in critiques
- Missing concepts
- Prioritized recommendations
- Cross-topic insights

**Use case**: Get actionable insights on what to study next.

## Workflow

### Typical Learning Session
1. **Start**: Use `get_progress()` to see current state
2. **Study**: Learn about a topic
3. **Evaluate**: Discuss with LLM, then call `save_evaluation()` to record results
4. **Visualize**: Use `plot_topic()` to see progress chart
5. **Analyze**: Use `analyze_gaps()` to get recommendations

### Voice Interaction Example
```
User: "Show me my progress"
LLM: [Calls get_progress()]

User: "Let's discuss gradient descent"
[Discussion happens...]

User: "Evaluate my understanding"
LLM: [Calls save_evaluation() with scores and explanations]

User: "Show me a chart of my progress"
LLM: [Calls plot_topic("gradient descent")]

User: "What should I focus on next?"
LLM: [Calls analyze_gaps()]
```

## Installation

### Local Development
```bash
git clone https://github.com/format37/checkpoint-mcp.git
cd checkpoint-mcp
./compose.local.sh
```

### Production Deployment
```bash
./compose.prod.sh
```

Note: Production deployment requires `.env.production` configuration.

## Data Storage
- Evaluations stored as CSV files in `/app/data/{topic}.csv`
- Visualizations saved as PNG in `/app/data/plots/`
- Each CSV contains: timestamp, answer, scores, and explanations

## Topics
See `backend/topics.py` for the complete list of available topics.
