# checkpoint-mcp
Knowledge evaluation MCP server

## Idea
I plan to develop MCP server to evaluate my knowledge.
To use it I would define topics that I need to learn and evaluation categories.
For example:
```
{
"gradient descent":
 {
 "width":
  {
  "evaluation":(float 0-1},
  "explanation":string
  },
 {
 "depth":
  {
  "evaluation":(float 0-1},
  "explanation":string
  }
}
```
Breadth refers to their ability to produce multiple valid solutions to a common ML problem, while depth refers to their ability to explain the details of one such approach.
I plan to talk with LLM in voice mode, maintaining the mock interview and then I would ask to call MCP to evaluate me. This would give me to see my learning progress and weakest points.

# Installation
To configure on remote server, need to configure the reverse-proxy.
```
git clone
cd
./compose.local
```
